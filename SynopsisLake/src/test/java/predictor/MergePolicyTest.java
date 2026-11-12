package predictor;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

public class MergePolicyTest {

	private static double[] doubles(String[] t, int from) {
		double[] d = new double[t.length - from];
		for (int i = 0; i < d.length; i++) {
			d[i] = Double.parseDouble(t[from + i]);
		}
		return d;
	}

	/** A model that scores a candidate by one feature (lower = dropped first): leaf 0 below the threshold, 1 above. */
	private static MergePolicy policyOn(String feature, double threshold) throws IOException {
		String model = "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=0\n"
				+ "objective=regression\nfeature_names=" + feature + "\n\nTree=0\nnum_leaves=2\nnum_cat=0\n"
				+ "split_feature=0\nthreshold=" + threshold + "\ndecision_type=2\nleft_child=-1\nright_child=-2\n"
				+ "leaf_value=0.0 1.0\nis_linear=0\n\nend of trees\n";
		return new MergePolicy(LightGBMModel.fromString(model));
	}

	@Test
	public void featuresMatchTheFixtureWrittenByPython() throws IOException {
		Map<String, double[]> vectors = new HashMap<String, double[]>();
		List<double[]> counts = new ArrayList<double[]>();
		List<String[]> expected = new ArrayList<String[]>();
		String[] names = null;
		try (InputStream in = getClass().getResourceAsStream("/merge_policy_golden.txt");
				BufferedReader r = new BufferedReader(new InputStreamReader(in))) {
			String line;
			while ((line = r.readLine()) != null) {
				String[] t = line.split(",");
				switch (t[0]) {
				case "counts":
					counts.add(doubles(t, 1));
					break;
				case "names":
					names = Arrays.copyOfRange(t, 1, t.length);
					break;
				case "feat":
					expected.add(t);
					break;
				default:
					vectors.put(t[0], doubles(t, 1));
				}
			}
		}
		assertArrayEquals(MergePolicy.FEATURE_NAMES, names);
		double[] lon = vectors.get("lon"), lat = vectors.get("lat");
		List<double[]> srcLon = new ArrayList<double[]>(), srcLat = new ArrayList<double[]>();
		// the fixture gives src_n per interior boundary; rebuild sources that coincide that many times
		srcLon.addAll(sourcesWithCoincidence(lon, vectors.get("src_lon")));
		srcLat.addAll(sourcesWithCoincidence(lat, vectors.get("src_lat")));
		MergePolicy.State s = new MergePolicy.State(lon, lat, counts.toArray(new double[0][]), srcLon, srcLat);
		for (int axis = 0; axis < 2; axis++) {
			double[] keep = vectors.get(axis == 0 ? "keep_lon" : "keep_lat");
			for (int i = 1; i < keep.length - 1; i++) {
				if (keep[i] == 0) {
					s.dropCanonical(axis, i);
				}
			}
		}
		int checked = 0;
		for (int axis = 0; axis < 2; axis++) {
			double[][] f = s.features(axis);
			List<String[]> rows = new ArrayList<String[]>();
			for (String[] e : expected) {
				if (Integer.parseInt(e[1]) == axis) {
					rows.add(e);
				}
			}
			assertEquals(rows.size(), f.length);
			for (int c = 0; c < f.length; c++) {
				double[] want = doubles(rows.get(c), 3);
				for (int k = 0; k < want.length; k++) {
					assertEquals("axis " + axis + " candidate " + c + " feature " + MergePolicy.FEATURE_NAMES[k],
							want[k], f[c][k], 1e-9 * Math.max(1.0, Math.abs(want[k])));
					checked++;
				}
			}
		}
		assertTrue(checked > 100);
	}

	/** Sources whose boundaries sit at the given interior boundaries as often as src_n says (ends always). */
	private static List<double[]> sourcesWithCoincidence(double[] canon, double[] srcNInterior) {
		int maxN = 0;
		for (double v : srcNInterior) {
			maxN = Math.max(maxN, (int) v);
		}
		List<double[]> sources = new ArrayList<double[]>();
		for (int s = 0; s < maxN; s++) {
			List<Double> b = new ArrayList<Double>();
			b.add(canon[0]);
			for (int i = 1; i < canon.length - 1; i++) {
				if (srcNInterior[i - 1] > s) {
					b.add(canon[i]);
				}
			}
			b.add(canon[canon.length - 1]);
			double[] arr = new double[b.size()];
			for (int i = 0; i < arr.length; i++) {
				arr[i] = b.get(i);
			}
			sources.add(arr);
		}
		return sources;
	}

	@Test
	public void coincidenceCountsSourcesWithABoundaryThere() {
		double[] canon = { 0, 1, 2, 3 };
		List<double[]> sources = new ArrayList<double[]>();
		sources.add(new double[] { 0, 1, 3 });
		sources.add(new double[] { 0, 2, 3 });
		sources.add(new double[] { 0, 1, 2, 3 });
		assertArrayEquals(new double[] { 3, 2, 2, 3 }, MergePolicy.coincidence(canon, sources), 0);
	}

	@Test
	public void dropsTheLowestScoringBoundaryFirstAndStopsAtTheTargets() throws IOException {
		double[] lon = { 0, 1, 2, 3, 4, 5, 6 };
		double[] lat = { 0, 1, 2 };
		double[][] counts = new double[6][2];
		for (double[] row : counts) {
			Arrays.fill(row, 50);
		}
		counts[3][0] = counts[3][1] = counts[4][0] = counts[4][1] = 0;      // buckets 3 and 4 are empty
		List<double[]> src = new ArrayList<double[]>();
		src.add(lon);
		List<double[]> srcLat = new ArrayList<double[]>();
		srcLat.add(lat);
		// merged mass log1p(cl+cr) below log1p(5): only the pair of empty buckets (3,4) scores 0, others 1
		MergePolicy p = policyOn("log_cm", Math.log1p(5));
		double[][] out = p.reshape(lon, lat, counts, src, srcLat, 5, 2);
		assertArrayEquals(new double[] { 0, 1, 2, 3, 5, 6 }, out[0], 0);     // boundary 4 (between 3 and 4) dropped
		assertArrayEquals(lat, out[1], 0);
		// two drops on lon, one on lat, always from the axis with most left
		double[][] out2 = p.reshape(lon, lat, counts, src, srcLat, 4, 1);
		assertEquals(5, out2[0].length);
		assertEquals(2, out2[1].length);
		assertEquals(0.0, out2[0][0], 0);
		assertEquals(6.0, out2[0][4], 0);
		// no reshaping when the targets are the aligned resolution
		assertArrayEquals(lon, p.reshape(lon, lat, counts, src, srcLat, 6, 2)[0], 0);
	}

	@Test
	public void rejectsBadTargetsAndUnknownFeatures() throws IOException {
		double[] lon = { 0, 1, 2 };
		double[] lat = { 0, 1 };
		double[][] counts = { { 1 }, { 1 } };
		List<double[]> a = new ArrayList<double[]>(), b = new ArrayList<double[]>();
		a.add(lon);
		b.add(lat);
		try {
			policyOn("log_cm", 1).reshape(lon, lat, counts, a, b, 3, 1);
			fail();
		} catch (IllegalArgumentException e) {
			assertTrue(e.getMessage().contains("aligned resolution"));
		}
		try {
			policyOn("not_a_feature", 1);
			fail();
		} catch (IllegalArgumentException e) {
			assertTrue(e.getMessage().contains("not_a_feature"));
		}
	}

	@Test
	public void massRuleMergesTheLightestPair() {
		double[] lon = { 0, 1, 2, 3, 4, 5 };
		double[] lat = { 0, 1 };
		double[][] counts = { { 40 }, { 30 }, { 1 }, { 2 }, { 50 } };
		List<double[]> a = new ArrayList<double[]>(), b = new ArrayList<double[]>();
		a.add(lon);
		b.add(lat);
		double[][] out = MergePolicy.massRule().reshape(lon, lat, counts, a, b, 4, 1);
		assertArrayEquals(new double[] { 0, 1, 2, 4, 5 }, out[0], 0);     // buckets 2 and 3 (mass 3) merged
	}

	@Test
	public void strategyFileSelectsArmMassOrLearned() throws IOException {
		java.io.File dir = java.nio.file.Files.createTempDirectory("strategy").toFile();
		assertEquals(null, writeAndLoad(dir, "strategy=arm\n").policy);
		assertEquals("mass", writeAndLoad(dir, "strategy=mass\n").name);
		for (String bad : new String[] { "strategy=best\n", "strategy=learned\n", "" }) {
			try {
				writeAndLoad(dir, bad);
				fail(bad);
			} catch (IllegalArgumentException e) {
				assertTrue(e.getMessage().contains("strategy"));
			}
		}
	}

	private static MergeStrategy writeAndLoad(java.io.File dir, String content) throws IOException {
		java.io.File f = new java.io.File(dir, "merge_strategy.properties");
		java.nio.file.Files.write(f.toPath(), content.getBytes("UTF-8"));
		return MergeStrategy.load(f.getPath());
	}
}
