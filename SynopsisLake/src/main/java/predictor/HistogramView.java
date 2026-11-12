package predictor;

import java.util.Arrays;

import histogram.SimpleSpatialHistogramOpt;

/**
 * Read-only snapshot of one histogram plus the per-histogram quantities the learned error features need
 * (cell areas, local density contrast, median density). Built once per synopsis, then reused for every query.
 * Counts are indexed [lon bucket][lat bucket], like SimpleSpatialHistogramOpt.getData().
 */
public class HistogramView {

	static final double E = 1e-9;

	final double[] bx;
	final double[] by;
	final double[] counts;
	final int nx;
	final int ny;
	final double[] area;
	/** |log(density / mean density of the 3x3 neighbourhood)| per cell */
	final double[] localContrast;
	final double medianDensity;
	final double total;

	public HistogramView(double[] lonBoundary, double[] latBoundary, double[][] data) {
		this.bx = lonBoundary.clone();
		this.by = latBoundary.clone();
		this.nx = bx.length - 1;
		this.ny = by.length - 1;
		this.counts = new double[nx * ny];
		this.area = new double[nx * ny];
		double[] density = new double[nx * ny];
		double sum = 0;
		int nonEmpty = 0;
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				int k = i * ny + j;
				counts[k] = data[i][j];
				area[k] = (bx[i + 1] - bx[i]) * (by[j + 1] - by[j]);
				density[k] = counts[k] / area[k];
				sum += counts[k];
				if (counts[k] > 0) {
					nonEmpty++;
				}
			}
		}
		this.total = sum;

		double[] nonEmptyDensity = new double[nonEmpty];
		int n = 0;
		for (int k = 0; k < counts.length; k++) {
			if (counts[k] > 0) {
				nonEmptyDensity[n++] = density[k];
			}
		}
		this.medianDensity = median(nonEmptyDensity);

		this.localContrast = new double[nx * ny];
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				double mean = 0;
				for (int di = -1; di <= 1; di++) {
					int ii = Math.min(Math.max(i + di, 0), nx - 1);
					for (int dj = -1; dj <= 1; dj++) {
						int jj = Math.min(Math.max(j + dj, 0), ny - 1);
						mean += density[ii * ny + jj];
					}
				}
				mean /= 9.0;
				localContrast[i * ny + j] = Math.abs(Math.log((density[i * ny + j] + E) / (mean + E)));
			}
		}
	}

	public static HistogramView of(SimpleSpatialHistogramOpt hist) {
		return new HistogramView(hist.getLonBoundary(), hist.getLatBoundary(), hist.getData());
	}

	public int lonBuckets() {
		return nx;
	}

	public double minLon() {
		return bx[0];
	}

	public double maxLon() {
		return bx[nx];
	}

	public double minLat() {
		return by[0];
	}

	public double maxLat() {
		return by[ny];
	}

	boolean overlaps(double qMinLon, double qMaxLon, double qMinLat, double qMaxLat) {
		return minLon() < qMaxLon && qMinLon < maxLon() && minLat() < qMaxLat && qMinLat < maxLat();
	}

	/** Same as numpy.median; 0 for an empty array. */
	static double median(double[] values) {
		if (values.length == 0) {
			return 0.0;
		}
		double[] s = values.clone();
		Arrays.sort(s);
		int m = s.length / 2;
		return s.length % 2 == 1 ? s[m] : 0.5 * (s[m - 1] + s[m]);
	}
}
