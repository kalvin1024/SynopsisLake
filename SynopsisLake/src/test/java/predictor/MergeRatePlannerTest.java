package predictor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

public class MergeRatePlannerTest {

	// exact count of [0,2]x[0,2] is 4 cells x 10 = 40; the 4x4 arm answers 40, the 2x2 arm answers `coarse`
	private static List<double[]> queries() {
		List<double[]> q = new ArrayList<double[]>();
		q.add(new double[] { 0, 2, 0, 2, 40 });
		q.add(new double[] { 0, 2, 0, 2, 40 });
		q.add(new double[] { 0, 1, 0, 1, 3 }); // exact count below minExact: skipped
		return q;
	}

	private static MergeRatePlanner.Plan plan(double coarse, double cutoff, double minCoverage) throws IOException {
		File dir = SynopsisStoreTest.store(coarse);
		return new MergeRatePlanner(0.10, 0.05, minCoverage, 10).plan(SynopsisStore.load(dir.getPath()),
				SynopsisStoreTest.PredictorFactoryTestSupport.model("area_cells"), CutoffTable.constant(cutoff),
				queries());
	}

	@Test
	public void picksSmallestSynopsisThatMeetsTheTarget() throws IOException {
		// 2x2 arm: 10 per cell as well -> [0,2]x[0,2] is one cell of 40 -> exact; both rates qualify
		MergeRatePlanner.Plan p = plan(40, 0.1, 0.5);
		assertEquals(0.25, p.chosenMergeRate, 0);
		assertEquals(2, p.rates.size());
		assertEquals(2, p.rates.get(0).queries); // the exact < 10 query is skipped
		assertEquals(1.0, p.rates.get(0).coverage, 0);
		assertTrue(p.rates.get(1).meetsTarget);
	}

	@Test
	public void aRateWithTooMuchErrorIsSkipped() throws IOException {
		MergeRatePlanner.Plan p = plan(100, 0.1, 0.5); // 2x2 arm answers 100 for an exact 40
		assertEquals(1.0, p.chosenMergeRate, 0);
		assertFalse(p.rates.get(1).meetsTarget);
		assertEquals(1.5, p.rates.get(1).meanRelErrAnswered, 1e-12);
		assertEquals(1.0, p.rates.get(1).violation, 0);
	}

	@Test
	public void noRateQualifiesWhenTheGateRejectsEverything() throws IOException {
		MergeRatePlanner.Plan p = plan(40, Double.NEGATIVE_INFINITY, 0.0);
		assertNull(p.chosenMergeRate);
		assertEquals(0.0, p.rates.get(0).coverage, 0);
		assertFalse(p.rates.get(0).meetsTarget);
	}

	@Test
	public void coverageBelowTheMinimumDisqualifiesARate() throws IOException {
		// score 0.05 is under cutoff 0.1 for every query, so coverage is 1; require more than possible
		assertNull(plan(40, 0.1, 1.5).chosenMergeRate);
	}

	@Test(expected = IllegalArgumentException.class)
	public void planningNeedsExactCounts() throws IOException {
		File dir = SynopsisStoreTest.store(40);
		List<double[]> q = new ArrayList<double[]>();
		q.add(new double[] { 0, 2, 0, 2, Double.NaN });
		new MergeRatePlanner(0.1, 0.05, 0, 10).plan(SynopsisStore.load(dir.getPath()),
				SynopsisStoreTest.PredictorFactoryTestSupport.model("area_cells"), CutoffTable.constant(1), q);
	}

	@Test
	public void queryFileHandlesMissingCounts() throws IOException {
		File f = File.createTempFile("queries", ".csv");
		f.deleteOnExit();
		java.nio.file.Files.write(f.toPath(), "0,1,0,1,5\n2,3,2,3\n\n".getBytes());
		List<double[]> q = QueryFile.load(f.getPath());
		assertEquals(2, q.size());
		assertEquals(5, q.get(0)[4], 0);
		assertTrue(Double.isNaN(q.get(1)[4]));
	}
}
