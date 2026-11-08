package predictor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import histogram.SimpleSpatialHistogramOpt;

/**
 * Reads the histogram files the logger writes (ml/SCHEMA.md, arm_&lt;lonRes&gt;.txt and src_&lt;i&gt;.txt): line 1 the merge
 * rate, line 2 the longitude boundaries, line 3 the latitude boundaries, then one line of counts per longitude
 * bucket. Numbers are written with full precision, so a loaded histogram answers exactly like the one that was
 * saved.
 */
public final class HistogramIO {

	private HistogramIO() {
	}

	public static final class Loaded {
		public final SimpleSpatialHistogramOpt hist;
		public final double mergeRate;

		Loaded(SimpleSpatialHistogramOpt hist, double mergeRate) {
			this.hist = hist;
			this.mergeRate = mergeRate;
		}
	}

	public static Loaded read(String path) throws IOException {
		try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
			String rateLine = reader.readLine();
			String lonLine = reader.readLine();
			String latLine = reader.readLine();
			if (rateLine == null || lonLine == null || latLine == null) {
				throw new IOException(path + ": not a histogram file (needs rate, lon and lat lines)");
			}
			double rate = Double.parseDouble(rateLine.trim());
			double[] lon = parse(lonLine);
			double[] lat = parse(latLine);
			List<double[]> rows = new ArrayList<double[]>();
			String line;
			while ((line = reader.readLine()) != null) {
				if (!line.trim().isEmpty()) {
					rows.add(parse(line));
				}
			}
			if (lon.length < 2 || lat.length < 2) {
				throw new IOException(path + ": needs at least one bucket per axis");
			}
			if (rows.size() != lon.length - 1) {
				throw new IOException(path + ": " + rows.size() + " count rows for " + (lon.length - 1)
						+ " longitude buckets");
			}
			double[][] data = new double[rows.size()][];
			for (int i = 0; i < data.length; i++) {
				data[i] = rows.get(i);
				if (data[i].length != lat.length - 1) {
					throw new IOException(path + ": row " + i + " has " + data[i].length + " counts for "
							+ (lat.length - 1) + " latitude buckets");
				}
			}
			SimpleSpatialHistogramOpt hist = new SimpleSpatialHistogramOpt(lon[0], lat[0], lon[lon.length - 1],
					lat[lat.length - 1], lon.length - 1, lat.length - 1, data, lon, lat);
			return new Loaded(hist, rate);
		}
	}

	private static double[] parse(String csv) {
		String[] p = csv.trim().split(",");
		double[] out = new double[p.length];
		for (int i = 0; i < p.length; i++) {
			out[i] = Double.parseDouble(p[i].trim());
		}
		return out;
	}
}
