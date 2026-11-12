package predictor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Learned replacement for ARM's reshape score: decides which boundaries of the aligned histogram to drop.
 *
 * The aligned histogram has every source histogram's boundaries. Reshaping to a target resolution drops
 * boundaries one at a time; dropping a boundary merges its two adjacent buckets. A candidate is scored by a
 * LightGBM model (merge_policy.txt, trained by ml/merge_policy.py) that predicts the log of the increase in
 * query error the drop would cause, so the lowest score is dropped first. The axis with the most drops left
 * (longitude on ties) is reduced first.
 *
 * Features describe the two buckets and the boundary only (no query workload is needed at build time). They must
 * give the same values as ml/merge_policy.py candidate_features; MergePolicyTest checks that against a fixture
 * written by `python ml/merge_policy.py golden`. The model file decides which of them it uses and in what order.
 */
public class MergePolicy {

	/** Every feature this class computes, in the order of ml/merge_policy.py FEATURES. */
	public static final String[] FEATURE_NAMES = { "axis", "progress", "pos", "log_wl", "log_wr", "log_w_ratio",
			"log_cl", "log_cr", "log_cm", "share", "dens_gap", "tv", "tv_w", "src_n", "log_dens_m" };

	private static final double SOURCE_TOLERANCE = 1e-9;

	private final LightGBMModel model;
	private final int[] column;

	public MergePolicy(LightGBMModel model) {
		this.model = model;
		String[] names = model.getFeatureNames();
		this.column = new int[names.length];
		for (int i = 0; i < names.length; i++) {
			int c = Arrays.asList(FEATURE_NAMES).indexOf(names[i]);
			if (c < 0) {
				throw new IllegalArgumentException("merge policy model uses unknown feature '" + names[i]
						+ "'; known: " + Arrays.toString(FEATURE_NAMES));
			}
			column[i] = c;
		}
	}

	public static MergePolicy load(String modelPath) throws IOException {
		return new MergePolicy(LightGBMModel.load(modelPath));
	}

	/**
	 * Boundaries of the reshaped histogram.
	 *
	 * @param lon, lat    boundaries of the aligned histogram (union of the sources' boundaries)
	 * @param counts      aligned counts, counts[lonBucket][latBucket]
	 * @param srcLon, srcLat boundaries of each source histogram (for the src_n feature)
	 * @param targetLon, targetLat buckets wanted per axis (at most the aligned resolution)
	 * @return {lonBoundaries, latBoundaries}, each a subset of the aligned boundaries including both ends
	 */
	public double[][] reshape(double[] lon, double[] lat, double[][] counts, List<double[]> srcLon,
			List<double[]> srcLat, int targetLon, int targetLat) {
		State s = new State(lon, lat, counts, srcLon, srcLat);
		if (targetLon < 1 || targetLat < 1 || targetLon > s.buckets(0) || targetLat > s.buckets(1)) {
			throw new IllegalArgumentException("targets " + targetLon + " x " + targetLat
					+ " must be between 1 and the aligned resolution " + s.buckets(0) + " x " + s.buckets(1));
		}
		int[] target = { targetLon, targetLat };
		while (true) {
			int remaining0 = s.buckets(0) - target[0];
			int remaining1 = s.buckets(1) - target[1];
			if (Math.max(remaining0, remaining1) <= 0) {
				break;
			}
			int axis = remaining0 >= remaining1 ? 0 : 1;
			double[][] f = s.features(axis);
			double[] x = new double[column.length];
			int best = 0;
			double bestScore = Double.POSITIVE_INFINITY;
			for (int c = 0; c < f.length; c++) {
				for (int i = 0; i < x.length; i++) {
					x[i] = f[c][column[i]];
				}
				double score = model.predict(x);
				if (score < bestScore) {
					bestScore = score;
					best = c;
				}
			}
			s.drop(axis, best);
		}
		return new double[][] { s.boundaries(0), s.boundaries(1) };
	}

	/** Number of sources with a boundary at each value of canon (within 1e-9). */
	static double[] coincidence(double[] canon, List<double[]> sources) {
		double[] n = new double[canon.length];
		for (int i = 0; i < canon.length; i++) {
			for (double[] src : sources) {
				int p = Arrays.binarySearch(src, canon[i]);
				if (p < 0) {
					p = -p - 1;
				}
				boolean hit = (p < src.length && Math.abs(src[p] - canon[i]) < SOURCE_TOLERANCE)
						|| (p > 0 && Math.abs(src[p - 1] - canon[i]) < SOURCE_TOLERANCE);
				n[i] += hit ? 1 : 0;
			}
		}
		return n;
	}

	/** The histogram being reshaped: current buckets plus which aligned boundaries are still kept. */
	static final class State {
		private final double[] canonBound0, canonBound1;
		private final double[] srcN0, srcN1;
		private final int[] n0;
		private List<Integer> kept0 = new ArrayList<Integer>();
		private List<Integer> kept1 = new ArrayList<Integer>();
		private double[][] cells;

		State(double[] lon, double[] lat, double[][] counts, List<double[]> srcLon, List<double[]> srcLat) {
			canonBound0 = lon;
			canonBound1 = lat;
			srcN0 = coincidence(lon, srcLon);
			srcN1 = coincidence(lat, srcLat);
			n0 = new int[] { lon.length - 1, lat.length - 1 };
			for (int i = 0; i < lon.length; i++) {
				kept0.add(i);
			}
			for (int i = 0; i < lat.length; i++) {
				kept1.add(i);
			}
			cells = new double[counts.length][];
			for (int i = 0; i < counts.length; i++) {
				cells[i] = counts[i].clone();
			}
		}

		int buckets(int axis) {
			return (axis == 0 ? kept0 : kept1).size() - 1;
		}

		double[] boundaries(int axis) {
			List<Integer> kept = axis == 0 ? kept0 : kept1;
			double[] canon = axis == 0 ? canonBound0 : canonBound1;
			double[] b = new double[kept.size()];
			for (int i = 0; i < b.length; i++) {
				b[i] = canon[kept.get(i)];
			}
			return b;
		}

		/** Drops candidate c (the c-th interior boundary) on the axis, merging buckets c and c + 1. */
		void drop(int axis, int c) {
			List<Integer> kept = axis == 0 ? kept0 : kept1;
			kept.remove(c + 1);
			if (axis == 0) {
				double[][] merged = new double[cells.length - 1][];
				for (int i = 0, k = 0; i < cells.length; i++) {
					if (i == c) {
						double[] row = new double[cells[i].length];
						for (int j = 0; j < row.length; j++) {
							row[j] = cells[i][j] + cells[i + 1][j];
						}
						merged[k++] = row;
						i++;
					} else {
						merged[k++] = cells[i];
					}
				}
				cells = merged;
			} else {
				for (int i = 0; i < cells.length; i++) {
					double[] row = new double[cells[i].length - 1];
					for (int j = 0, k = 0; j < cells[i].length; j++) {
						if (j == c) {
							row[k++] = cells[i][j] + cells[i][j + 1];
							j++;
						} else {
							row[k++] = cells[i][j];
						}
					}
					cells[i] = row;
				}
			}
		}

		/** Drops a boundary given by its index in the aligned boundaries (used to set up a state in tests). */
		void dropCanonical(int axis, int canonicalIndex) {
			List<Integer> kept = axis == 0 ? kept0 : kept1;
			int pos = kept.indexOf(canonicalIndex);
			if (pos < 1 || pos > kept.size() - 2) {
				throw new IllegalArgumentException("not an interior kept boundary: " + canonicalIndex);
			}
			drop(axis, pos - 1);
		}

		/** One row per candidate on the axis, columns in FEATURE_NAMES order. */
		double[][] features(int axis) {
			List<Integer> kept = axis == 0 ? kept0 : kept1;
			double[] canon = axis == 0 ? canonBound0 : canonBound1;
			double[] srcN = axis == 0 ? srcN0 : srcN1;
			double[][] m = axis == 0 ? cells : transpose(cells);
			int n = kept.size() - 1;
			int width = m[0].length;
			double[] w = new double[n];
			double[] rowSum = new double[n];
			double total = 0;
			for (int i = 0; i < n; i++) {
				w[i] = canon[kept.get(i + 1)] - canon[kept.get(i)];
				double sum = 0;
				for (int k = 0; k < width; k++) {
					sum += m[i][k];
				}
				rowSum[i] = sum;
				total += sum;
			}
			double[][] f = new double[n - 1][FEATURE_NAMES.length];
			for (int j = 0; j < n - 1; j++) {
				double cl = rowSum[j], cr = rowSum[j + 1], wl = w[j], wr = w[j + 1];
				double dl = cl / wl, dr = cr / wr;
				double tv = 0;
				for (int k = 0; k < width; k++) {
					tv += Math.abs(m[j][k] / (cl + 1e-9) - m[j + 1][k] / (cr + 1e-9));
				}
				tv *= 0.5;
				double[] row = f[j];
				row[0] = axis;
				row[1] = (double) n / n0[axis];
				row[2] = (double) j / n;
				row[3] = Math.log(wl);
				row[4] = Math.log(wr);
				row[5] = Math.log(wl / wr);
				row[6] = Math.log1p(cl);
				row[7] = Math.log1p(cr);
				row[8] = Math.log1p(cl + cr);
				row[9] = (cl + cr) / total;
				row[10] = Math.abs(Math.log1p(dl) - Math.log1p(dr));
				row[11] = tv;
				row[12] = tv * Math.log1p(Math.min(cl, cr));
				row[13] = srcN[kept.get(j + 1)];
				row[14] = Math.log1p((cl + cr) / (wl + wr));
			}
			return f;
		}

		private static double[][] transpose(double[][] a) {
			double[][] t = new double[a[0].length][a.length];
			for (int i = 0; i < a.length; i++) {
				for (int j = 0; j < a[i].length; j++) {
					t[j][i] = a[i][j];
				}
			}
			return t;
		}
	}
}
