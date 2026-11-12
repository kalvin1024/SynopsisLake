package predictor;

/**
 * What one range query looks like on one histogram: the building blocks of the learned error features.
 * Ported from ml/features.py (_overlap, _query_features); the two must stay in step.
 */
public class HistogramQueryStats {

	public final double est;
	public final double areaCells;
	public final int nCells;
	public final int nPartial;
	public final double sideFrac;
	/** count in the fully covered cells */
	public final double fullCount;
	/** count in every touched cell, i.e. the largest possible answer */
	public final double touchedCount;
	public final double edgeContrast;
	public final double inOut;
	public final double emptyPartial;
	public final double estFrac;
	public final double densityRatio;

	private HistogramQueryStats(double est, double areaCells, int nCells, int nPartial, double sideFrac,
			double fullCount, double touchedCount, double edgeContrast, double inOut, double emptyPartial,
			double estFrac, double densityRatio) {
		this.est = est;
		this.areaCells = areaCells;
		this.nCells = nCells;
		this.nPartial = nPartial;
		this.sideFrac = sideFrac;
		this.fullCount = fullCount;
		this.touchedCount = touchedCount;
		this.edgeContrast = edgeContrast;
		this.inOut = inOut;
		this.emptyPartial = emptyPartial;
		this.estFrac = estFrac;
		this.densityRatio = densityRatio;
	}

	public static HistogramQueryStats compute(HistogramView h, double a, double b, double c, double d) {
		final double E = HistogramView.E;
		int[] ix = span(h.bx, a, b);
		int[] iy = span(h.by, c, d);
		int i0 = ix[0], i1 = ix[1], j0 = iy[0], j1 = iy[1];
		double[] fx = fractions(h.bx, i0, i1, a, b);
		double[] fy = fractions(h.by, j0, j1, c, d);

		double est = 0, wSum = 0, fullCount = 0, touchedCount = 0, subSum = 0, aIn = 0;
		double contrastNum = 0, contrastDen = 0;
		int nCells = 0, nPartial = 0, emptyPartial = 0;
		for (int i = i0; i <= i1; i++) {
			for (int j = j0; j <= j1; j++) {
				int k = i * h.ny + j;
				double w = fx[i - i0] * fy[j - j0];
				double cnt = h.counts[k];
				subSum += cnt;
				aIn += h.area[k];
				est += w * cnt;
				wSum += w;
				if (w > 0) {
					nCells++;
					touchedCount += cnt;
				}
				if (w >= 1 - 1e-12) {
					fullCount += cnt;
				} else if (w > 0) {
					nPartial++;
					double pw = cnt + 1;
					contrastNum += h.localContrast[k] * pw;
					contrastDen += pw;
					if (cnt == 0) {
						emptyPartial++;
					}
				}
			}
		}
		double edgeContrast = nPartial > 0 ? contrastNum / contrastDen : 0.0;

		int bi0 = Math.max(i0 - 1, 0), bi1 = Math.min(i1 + 2, h.nx);
		int bj0 = Math.max(j0 - 1, 0), bj1 = Math.min(j1 + 2, h.ny);
		double blockCount = 0, blockArea = 0;
		for (int i = bi0; i < bi1; i++) {
			for (int j = bj0; j < bj1; j++) {
				blockCount += h.counts[i * h.ny + j];
				blockArea += h.area[i * h.ny + j];
			}
		}
		double ringCount = blockCount - subSum;
		double ringArea = blockArea - aIn;
		double dIn = subSum / aIn;
		double dOut = ringArea > 0 ? ringCount / ringArea : dIn;

		return new HistogramQueryStats(est, wSum, nCells, nPartial, (b - a) / (h.bx[h.nx] - h.bx[0]), fullCount,
				touchedCount, edgeContrast, Math.log((dIn + E) / (dOut + E)),
				nPartial > 0 ? (double) emptyPartial / nPartial : 0.0, est / Math.max(h.total, 1.0),
				Math.log((dIn + E) / (h.medianDensity + E)));
	}

	/** (max possible - min possible) / max(est, 1) */
	public double boundRel() {
		return (touchedCount - fullCount) / Math.max(est, 1.0);
	}

	public double edgeShare() {
		return (est - fullCount) / Math.max(est, HistogramView.E);
	}

	/** First and last bucket the interval touches, clipped to the histogram (numpy searchsorted right/left). */
	private static int[] span(double[] b, double lo, double hi) {
		int n = b.length - 1;
		int i0 = clip(upperBound(b, lo) - 1, 0, n - 1);
		int i1 = clip(lowerBound(b, hi) - 1, 0, n - 1);
		return new int[] { i0, i1 };
	}

	private static double[] fractions(double[] b, int i0, int i1, double lo, double hi) {
		double[] f = new double[i1 - i0 + 1];
		for (int i = i0; i <= i1; i++) {
			double left = b[i], right = b[i + 1];
			double frac = (Math.min(hi, right) - Math.max(lo, left)) / (right - left);
			f[i - i0] = Math.min(Math.max(frac, 0.0), 1.0);
		}
		return f;
	}

	/** index of the first element > v */
	private static int upperBound(double[] a, double v) {
		int lo = 0, hi = a.length;
		while (lo < hi) {
			int mid = (lo + hi) >>> 1;
			if (a[mid] <= v) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}
		return lo;
	}

	/** index of the first element >= v */
	private static int lowerBound(double[] a, double v) {
		int lo = 0, hi = a.length;
		while (lo < hi) {
			int mid = (lo + hi) >>> 1;
			if (a[mid] < v) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}
		return lo;
	}

	private static int clip(int v, int lo, int hi) {
		return Math.min(Math.max(v, lo), hi);
	}
}
