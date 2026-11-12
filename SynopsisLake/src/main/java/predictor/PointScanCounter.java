package predictor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/** Exact count by scanning every point (the cost the gate is trying to avoid). Bounds are inclusive. */
public class PointScanCounter implements RawRangeCounter {

	private double[] xs = new double[1024];
	private double[] ys = new double[1024];
	private int n;

	/** Reads tab-separated "lon lat" lines, the format of the data files. */
	public static PointScanCounter fromFiles(List<String> paths) throws IOException {
		PointScanCounter c = new PointScanCounter();
		for (String path : paths) {
			try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
				String line;
				while ((line = reader.readLine()) != null) {
					String[] t = line.split("\t");
					c.add(Double.parseDouble(t[0]), Double.parseDouble(t[1]));
				}
			}
		}
		return c;
	}

	public void add(double lon, double lat) {
		if (n == xs.length) {
			xs = Arrays.copyOf(xs, n * 2);
			ys = Arrays.copyOf(ys, n * 2);
		}
		xs[n] = lon;
		ys[n] = lat;
		n++;
	}

	public int size() {
		return n;
	}

	@Override
	public double count(double minLon, double maxLon, double minLat, double maxLat) {
		int c = 0;
		for (int i = 0; i < n; i++) {
			if (xs[i] >= minLon && xs[i] <= maxLon && ys[i] >= minLat && ys[i] <= maxLat) {
				c++;
			}
		}
		return c;
	}
}
