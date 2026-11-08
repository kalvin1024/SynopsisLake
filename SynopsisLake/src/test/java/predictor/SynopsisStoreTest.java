package predictor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import org.junit.Test;

import histogram.SimpleSpatialHistogramOpt;

public class SynopsisStoreTest {

	/** Writes a histogram file in the logger's format: rate, lon boundaries, lat boundaries, count rows. */
	static void write(File f, double rate, double[] lon, double[] lat, double[][] data) throws IOException {
		StringBuilder sb = new StringBuilder();
		sb.append(rate).append('\n').append(join(lon)).append('\n').append(join(lat)).append('\n');
		for (double[] row : data) {
			sb.append(join(row)).append('\n');
		}
		Files.write(f.toPath(), sb.toString().getBytes());
	}

	private static String join(double[] v) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < v.length; i++) {
			sb.append(i > 0 ? "," : "").append(v[i]);
		}
		return sb.toString();
	}

	static double[][] filled(int lon, int lat, double v) {
		double[][] d = new double[lon][lat];
		for (double[] row : d) {
			Arrays.fill(row, v);
		}
		return d;
	}

	static double[] bounds(int n) {
		double[] b = new double[n + 1];
		for (int i = 0; i <= n; i++) {
			b[i] = 4.0 * i / n;
		}
		return b;
	}

	/** src: 4x4 with 10/cell. arms: 4x4 at rate 1.0 and 2x2 at rate 0.25, each cell holding `coarse`. */
	static File store(double coarse) throws IOException {
		File dir = java.nio.file.Files.createTempDirectory("store").toFile();
		dir.deleteOnExit();
		write(new File(dir, "src_0.txt"), 1.0, bounds(4), bounds(4), filled(4, 4, 10));
		write(new File(dir, "arm_4.txt"), 1.0, bounds(4), bounds(4), filled(4, 4, 10));
		write(new File(dir, "arm_2.txt"), 0.25, bounds(2), bounds(2), filled(2, 2, coarse));
		return dir;
	}

	@Test
	public void loadedHistogramAnswersLikeTheOriginal() throws IOException {
		File dir = store(40);
		HistogramIO.Loaded l = HistogramIO.read(new File(dir, "arm_4.txt").getPath());
		SimpleSpatialHistogramOpt orig = new SimpleSpatialHistogramOpt(0, 0, 4, 4, 4, 4);
		for (double[] row : orig.getData()) {
			Arrays.fill(row, 10);
		}
		assertEquals(1.0, l.mergeRate, 0);
		double[][] qs = { { 0.5, 1.5, 0, 1 }, { 0, 4, 0, 4 }, { 1.2, 3.3, 0.7, 2.9 }, { 10, 11, 10, 11 } };
		for (double[] q : qs) {
			assertEquals(orig.RangeQuery(q[0], q[1], q[2], q[3]), l.hist.RangeQuery(q[0], q[1], q[2], q[3]), 1e-12);
		}
	}

	@Test
	public void storeListsRatesLargestFirstAndFindsNearest() throws IOException {
		SynopsisStore s = SynopsisStore.load(store(40).getPath());
		List<Double> rates = s.mergeRates();
		assertEquals(Arrays.asList(1.0, 0.25), rates);
		assertEquals(0.25, s.nearestMergeRate(0.3), 0);
		assertEquals(1, s.numSources());
		assertNull(s.context(1.0, false).mergedBeta);
		assertNotNull(s.context(1.0, true).mergedBeta);
		try {
			s.merged(0.5);
			fail();
		} catch (IllegalArgumentException e) {
			assertTrue(e.getMessage().contains("0.5"));
		}
	}

	@Test
	public void malformedFilesAreRejected() throws IOException {
		File f = File.createTempFile("bad", ".txt");
		f.deleteOnExit();
		Files.write(f.toPath(), "1.0\n0,1,2\n0,1\n5,5\n".getBytes()); // 2 lon buckets, 1 count row
		try {
			HistogramIO.read(f.getPath());
			fail();
		} catch (IOException e) {
			assertTrue(e.getMessage().contains("longitude buckets"));
		}
		File empty = java.nio.file.Files.createTempDirectory("empty").toFile();
		try {
			SynopsisStore.load(empty.getPath());
			fail();
		} catch (IOException e) {
			assertTrue(e.getMessage().contains("src_"));
		}
	}

	@Test
	public void needsBetaFollowsTheModel() throws IOException {
		assertTrue(new SkewnessAlignPredictor().needsBeta());
		assertFalse(PredictorFactoryTestSupport.model("area_cells").needsBeta());
		assertTrue(PredictorFactoryTestSupport.model("skew_pred").needsBeta());
		assertTrue(PredictorFactoryTestSupport.model("log_alpha").needsBeta());
	}

	/** A one-split model on a named feature, identity label. */
	static final class PredictorFactoryTestSupport {
		static LearnedErrorPredictor model(String feature) throws IOException {
			return model(feature, 1e9, 0.05, 0.05);
		}

		static LearnedErrorPredictor model(String feature, double threshold, double left, double right)
				throws IOException {
			String text = "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=0\n"
					+ "objective=regression\nfeature_names=" + feature + "\n\nTree=0\nnum_leaves=2\nnum_cat=0\n"
					+ "split_feature=0\nthreshold=" + threshold + "\ndecision_type=2\nleft_child=-1\n"
					+ "right_child=-2\nleaf_value=" + left + " " + right + "\nis_linear=0\n\nend of trees\n";
			Properties p = new Properties();
			p.setProperty(LearnedErrorPredictor.LABEL_TRANSFORM, "identity");
			return new LearnedErrorPredictor(LightGBMModel.fromString(text), p);
		}
	}
}
