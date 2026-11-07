package predictor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.Properties;

import org.apache.commons.math3.distribution.BetaDistribution;
import org.junit.Test;

import histogram.SimpleSpatialHistogramOpt;

public class LearnedErrorPredictorTest {

	private static final BetaDistribution BETA = new BetaDistribution(2, 2);

	/** 4x4 histogram over [0,4]x[0,4], 10 points in every cell. */
	private static SimpleSpatialHistogramOpt grid() {
		SimpleSpatialHistogramOpt h = new SimpleSpatialHistogramOpt(0, 0, 4, 4, 4, 4);
		for (double[] row : h.getData()) {
			java.util.Arrays.fill(row, 10);
		}
		return h;
	}

	private static SynopsisContext context(BetaDistribution beta) {
		SimpleSpatialHistogramOpt merged = grid();
		return new SynopsisContext(merged, new SimpleSpatialHistogramOpt[] { grid() }, 0.5, beta);
	}

	/** One split: left leaf if `feature` <= threshold, else right leaf. */
	private static String stump(String feature, double threshold, double left, double right) {
		return "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=0\n"
				+ "objective=regression\nfeature_names=" + feature + "\n\nTree=0\nnum_leaves=2\nnum_cat=0\n"
				+ "split_feature=0\nthreshold=" + threshold + "\ndecision_type=2\nleft_child=-1\nright_child=-2\n"
				+ "leaf_value=" + left + " " + right + "\nis_linear=0\n\nend of trees\n";
	}

	private static LearnedErrorPredictor predictor(String modelText, Properties config) throws IOException {
		return new LearnedErrorPredictor(LightGBMModel.fromString(modelText), config);
	}

	// query [0.5,1.5] x [0,1] covers half of two cells -> est = 10, log1p(est) = 2.398
	private static final double[] Q = { 0.5, 1.5, 0, 1 };

	@Test
	public void logLabelIsExponentiated() throws IOException {
		LearnedErrorPredictor p = predictor(stump("log_est", 2.0, 0.1, 0.25), new Properties());
		double got = p.bind(context(null)).estimate(Q[0], Q[1], Q[2], Q[3]);
		assertEquals(Math.exp(0.25), got, 1e-12); // log1p(10) > 2.0 -> right leaf
	}

	@Test
	public void identityLabelIsNotTransformed() throws IOException {
		Properties config = new Properties();
		config.setProperty(LearnedErrorPredictor.LABEL_TRANSFORM, "identity");
		LearnedErrorPredictor p = predictor(stump("log_est", 5.0, 0.1, 0.25), config);
		assertEquals(0.1, p.bind(context(null)).estimate(Q[0], Q[1], Q[2], Q[3]), 1e-12);
	}

	@Test
	public void usesWhicheverFeaturesTheModelNames() throws IOException {
		// merge_rate = 0.5 comes from the context, n_src_overlap from the source histograms
		Properties id = new Properties();
		id.setProperty(LearnedErrorPredictor.LABEL_TRANSFORM, "identity");
		assertEquals(7, predictor(stump("merge_rate", 0.6, 7, 9), id).bind(context(null)).estimate(0.5, 1.5, 0, 1), 0);
		assertEquals(9, predictor(stump("n_src_overlap", 0.5, 7, 9), id).bind(context(null)).estimate(0.5, 1.5, 0, 1),
				0);
		// a query outside every source overlaps none of them
		assertEquals(7, predictor(stump("n_src_overlap", 0.5, 7, 9), id).bind(context(null)).estimate(10, 11, 10, 11),
				0);
	}

	@Test
	public void skewnessAlignInputNeedsBetaOnlyWhenUsed() throws IOException {
		Properties id = new Properties();
		id.setProperty(LearnedErrorPredictor.LABEL_TRANSFORM, "identity");
		// no beta, model does not use skew_pred: fine
		predictor(stump("log_est", 2.0, 1, 2), id).bind(context(null)).estimate(0.5, 1.5, 0, 1);
		// with beta, skew_pred is the histogram's own Skewness-Align value
		SynopsisContext ctx = context(BETA);
		double expected = ctx.mergedHist.RangeQueryWithRatio(BETA, 0.5, 1.5, 0, 1)[2];
		assertEquals(expected, new SkewnessAlignPredictor().bind(ctx).estimate(0.5, 1.5, 0, 1), 0);
		double thr = expected - 1e-9;
		assertEquals(2, predictor(stump("skew_pred", thr, 1, 2), id).bind(ctx).estimate(0.5, 1.5, 0, 1), 0);
		// model uses skew_pred but no beta was given
		try {
			predictor(stump("skew_pred", 0.1, 1, 2), id).bind(context(null)).estimate(0.5, 1.5, 0, 1);
			fail();
		} catch (IllegalStateException expectedError) {
			assertTrue(expectedError.getMessage().contains("beta"));
		}
	}

	@Test
	public void unknownFeatureFailsAtLoad() {
		try {
			predictor(stump("not_a_feature", 0.5, 1, 2), new Properties());
			fail();
		} catch (IllegalArgumentException | IOException e) {
			assertTrue(e.getMessage().contains("not_a_feature"));
		}
	}

	@Test
	public void customFeatureCanBeRegistered() throws IOException {
		FeatureRegistry.register("test_const_three", q -> 3.0);
		Properties id = new Properties();
		id.setProperty(LearnedErrorPredictor.LABEL_TRANSFORM, "identity");
		assertEquals(2, predictor(stump("test_const_three", 2.5, 1, 2), id).bind(context(null)).estimate(0, 1, 0, 1),
				0);
	}

	@Test
	public void gateFollowsCutoff() throws IOException {
		Properties config = new Properties();
		assertNull(predictor(stump("log_est", 2.0, 0.1, 0.25), config).getGate());
		config.setProperty(LearnedErrorPredictor.GATE_CUTOFF, "0.5");
		ErrorGate gate = predictor(stump("log_est", 2.0, 0.1, 0.25), config).getGate();
		assertTrue(gate.useSynopsis(0.5));
		assertTrue(!gate.useSynopsis(0.51));
	}
}
