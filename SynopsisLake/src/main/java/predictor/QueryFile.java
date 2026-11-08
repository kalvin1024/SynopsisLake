package predictor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Range queries, one per line: minLon,maxLon,minLat,maxLat[,exactCount]. Each query is returned as
 * {minLon, maxLon, minLat, maxLat, exact}, with exact NaN when the file has no count column.
 */
public final class QueryFile {

	private QueryFile() {
	}

	public static List<double[]> load(String path) throws IOException {
		List<double[]> queries = new ArrayList<double[]>();
		try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
			String line;
			int lineNo = 0;
			while ((line = reader.readLine()) != null) {
				lineNo++;
				if (line.trim().isEmpty()) {
					continue;
				}
				String[] p = line.trim().split(",");
				if (p.length < 4) {
					throw new IOException(path + ":" + lineNo + ": needs minLon,maxLon,minLat,maxLat[,exact]");
				}
				double[] q = new double[5];
				for (int i = 0; i < 4; i++) {
					q[i] = Double.parseDouble(p[i]);
				}
				q[4] = p.length > 4 ? Double.parseDouble(p[4]) : Double.NaN;
				queries.add(q);
			}
		}
		return queries;
	}
}
