package predictor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Properties;

import org.junit.Test;

import histogram.SimpleSpatialHistogramOpt;

public class GatedRangeQueryTest {

	/** 4x4 histogram over [0,4]x[0,4], 10 points in every cell. */
	private static SimpleSpatialHistogramOpt grid() {
		SimpleSpatialHistogramOpt h = new SimpleSpatialHistogramOpt(0, 0, 4, 4, 4, 4);
		for (double[] row : h.getData()) {
			java.util.Arrays.fill(row, 10);
		}
		return h;
	}

	private static SynopsisContext context() {
		return new SynopsisContext(grid(), new SimpleSpatialHistogramOpt[] { grid() }, 0.5, null);
	}

	/** Predictor whose estimate is the query's n_cells feature, as a relative error. */
	private static ErrorPredictor predictorOnNCells() throws IOException {
		String model = "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=0\n"
				+ "objective=regression\nfeature_names=area_cells\n\nTree=0\nnum_leaves=2\nnum_cat=0\n"
				+ "split_feature=0\nthreshold=1.5\ndecision_type=2\nleft_child=-1\nright_child=-2\n"
				+ "leaf_value=0.05 0.4\nis_linear=0\n\nend of trees\n";
		Properties p = new Properties();
		p.setProperty(LearnedErrorPredictor.LABEL_TRANSFORM, "identity");
		return new LearnedErrorPredictor(LightGBMModel.fromString(model), p);
	}

	private static PointScanCounter points(int perCell) {
		PointScanCounter c = new PointScanCounter();
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				for (int k = 0; k < perCell; k++) {
					c.add(i + 0.5, j + 0.5);
				}
			}
		}
		return c;
	}

	@Test
	public void smallQueryIsAnsweredFromSynopsisWithItsErrorEstimate() throws IOException {
		GatedRangeQuery g = new GatedRangeQuery(context(), predictorOnNCells(), 0.1, points(12));
		ApproxAnswer a = g.answer(0.5, 1.0, 0.5, 1.0); // inside one cell
		assertTrue(a.fromSynopsis);
		assertEquals(0.05, a.estimatedRelError, 1e-12);
		assertEquals(a.synopsisEstimate, a.answer, 0);
		assertEquals(0, a.rawNanos);
	}

	@Test
	public void untrustedQueryGoesToRawData() throws IOException {
		GatedRangeQuery g = new GatedRangeQuery(context(), predictorOnNCells(), 0.1, points(12));
		ApproxAnswer a = g.answer(0, 4, 0, 4);
		assertFalse(a.fromSynopsis);
		assertEquals(0.4, a.errorScore, 1e-12);
		assertEquals(16 * 12, a.answer, 0); // exact, not the synopsis's 160
		assertEquals(160, a.synopsisEstimate, 1e-9);
		assertEquals(0.0, a.estimatedRelError, 0);
		assertTrue(a.rawNanos > 0);
	}

	@Test
	public void withoutRawDataTheDecisionIsStillReported() throws IOException {
		GatedRangeQuery g = new GatedRangeQuery(context(), predictorOnNCells(), 0.1, null);
		ApproxAnswer a = g.answer(0, 4, 0, 4);
		assertFalse(a.fromSynopsis);
		assertEquals(a.synopsisEstimate, a.answer, 0);
		assertEquals(0.4, a.estimatedRelError, 1e-12); // the answer is the synopsis estimate: its error estimate
	}

	@Test
	public void skewnessAlignScoreIsNotReportedAsRelativeError() {
		SynopsisContext ctx = new SynopsisContext(grid(), new SimpleSpatialHistogramOpt[] { grid() }, 0.5,
				new org.apache.commons.math3.distribution.BetaDistribution(2, 2));
		ApproxAnswer a = new GatedRangeQuery(ctx, new SkewnessAlignPredictor(), Double.POSITIVE_INFINITY, null)
				.answer(0.5, 1.5, 0, 1);
		assertTrue(a.fromSynopsis);
		assertTrue(Double.isNaN(a.estimatedRelError));
	}

	@Test
	public void cutoffTablePerMergeRate() throws IOException {
		File f = File.createTempFile("cutoffs", ".csv");
		f.deleteOnExit();
		Files.write(f.toPath(), ("mergeRate,eps,delta,cutoff\n1.0,0.1,0.05,0.2\n0.5,0.1,0.05,-inf\n"
				+ "1.0,0.25,0.05,0.4\n").getBytes());
		CutoffTable t = CutoffTable.load(f.getPath(), 0.1, 0.05, null);
		assertEquals(0.2, t.forMergeRate(1.0), 0);
		assertEquals(Double.NEGATIVE_INFINITY, t.forMergeRate(0.5 + 1e-9), 0);
		assertEquals(0.4, CutoffTable.load(f.getPath(), 0.25, 0.05, null).forMergeRate(1.0), 0);
		try {
			t.forMergeRate(0.3);
			fail();
		} catch (IllegalStateException e) {
			assertTrue(e.getMessage().contains("0.3"));
		}
		assertEquals(0.9, CutoffTable.load(f.getPath(), 0.1, 0.05, 0.9).forMergeRate(0.3), 0);
		// a file with several rows and no mergeRate column is ambiguous
		Files.write(f.toPath(), "cutoff\n1\n2\n".getBytes());
		try {
			CutoffTable.load(f.getPath(), null, null, null);
			fail();
		} catch (IOException e) {
			assertTrue(e.getMessage().contains("one row"));
		}
	}

	@Test
	public void cutoffsComeFromFileThenValueThenModelProperties() throws IOException {
		Properties p = new Properties();
		p.setProperty(LearnedErrorPredictor.LABEL_TRANSFORM, "identity");
		p.setProperty(LearnedErrorPredictor.GATE_CUTOFF, "0.3");
		ErrorPredictor withProps = new LearnedErrorPredictor(
				LightGBMModel.fromString(modelText()), p);
		assertEquals(0.3, PredictorFactory.cutoffs(null, null, null, null, withProps).forMergeRate(1.0), 0);
		assertEquals(0.7, PredictorFactory.cutoffs(null, 0.7, null, null, withProps).forMergeRate(1.0), 0);
		assertEquals(null, PredictorFactory.cutoffs(null, null, null, null, new SkewnessAlignPredictor()));
		assertTrue(PredictorFactory.predictor(null, null) instanceof SkewnessAlignPredictor);
	}

	@Test
	public void infiniteGateCutoffInPropertiesIsUnderstood() throws IOException {
		Properties p = new Properties();
		p.setProperty(LearnedErrorPredictor.GATE_CUTOFF, "-inf");
		LearnedErrorPredictor lp = new LearnedErrorPredictor(LightGBMModel.fromString(modelText()), p);
		assertFalse(lp.getGate().useSynopsis(0.0));
	}

	private static String modelText() {
		return "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=0\n"
				+ "objective=regression\nfeature_names=area_cells\n\nTree=0\nnum_leaves=2\nnum_cat=0\n"
				+ "split_feature=0\nthreshold=1.5\ndecision_type=2\nleft_child=-1\nright_child=-2\n"
				+ "leaf_value=0.05 0.4\nis_linear=0\n\nend of trees\n";
	}
}
