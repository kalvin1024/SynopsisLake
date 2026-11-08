package predictor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Gate cutoffs per merge rate, calibrated offline (ml/gate_eval.cutoff, one cutoff per merge rate).
 *
 * File format (cutoffs.csv): a header line, then one row per merge rate. Columns are found by name:
 *   cutoff     required; a number, or inf / -inf (-inf sends every query to the raw data)
 *   mergeRate  optional; without it the file must have exactly one row, which applies to every merge rate
 *   eps, delta optional; when present, rows are selected for the requested eps and delta
 *
 * A merge rate that has no row falls back to the default cutoff if one was given, and is an error otherwise.
 */
public class CutoffTable {

	private static final double RATE_TOLERANCE = 1e-6;

	private final double[] rates;
	private final double[] cutoffs;
	private final Double fallback;

	private CutoffTable(double[] rates, double[] cutoffs, Double fallback) {
		this.rates = rates;
		this.cutoffs = cutoffs;
		this.fallback = fallback;
	}

	/** One cutoff for every merge rate. */
	public static CutoffTable constant(double cutoff) {
		return new CutoffTable(new double[0], new double[0], cutoff);
	}

	/**
	 * @param eps      error limit to select rows for, or null to not filter
	 * @param delta    violation rate to select rows for, or null to not filter
	 * @param fallback cutoff for merge rates without a row, or null for none
	 */
	public static CutoffTable load(String path, Double eps, Double delta, Double fallback) throws IOException {
		List<double[]> rows = new ArrayList<double[]>();
		try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
			String header = reader.readLine();
			if (header == null) {
				throw new IOException("empty cutoffs file " + path);
			}
			List<String> cols = Arrays.asList(header.trim().split(","));
			int cutoffCol = cols.indexOf("cutoff");
			if (cutoffCol < 0) {
				throw new IOException(path + ": needs a 'cutoff' column, found " + cols);
			}
			int rateCol = cols.indexOf("mergeRate");
			int epsCol = cols.indexOf("eps");
			int deltaCol = cols.indexOf("delta");
			String line;
			while ((line = reader.readLine()) != null) {
				if (line.trim().isEmpty()) {
					continue;
				}
				String[] p = line.trim().split(",");
				if (eps != null && epsCol >= 0 && Math.abs(Double.parseDouble(p[epsCol]) - eps) > 1e-9) {
					continue;
				}
				if (delta != null && deltaCol >= 0 && Math.abs(Double.parseDouble(p[deltaCol]) - delta) > 1e-9) {
					continue;
				}
				rows.add(new double[] { rateCol < 0 ? Double.NaN : Double.parseDouble(p[rateCol]),
						parseNumber(p[cutoffCol]) });
			}
		}
		if (rows.isEmpty()) {
			throw new IOException(path + ": no rows for eps=" + eps + " delta=" + delta);
		}
		if (rows.get(0)[0] != rows.get(0)[0]) { // no mergeRate column
			if (rows.size() != 1) {
				throw new IOException(path + ": without a mergeRate column the file must have one row, has "
						+ rows.size() + " (add eps/delta to pick one?)");
			}
			return constant(rows.get(0)[1]);
		}
		double[] rates = new double[rows.size()];
		double[] cutoffs = new double[rows.size()];
		for (int i = 0; i < rates.length; i++) {
			rates[i] = rows.get(i)[0];
			cutoffs[i] = rows.get(i)[1];
			for (int j = 0; j < i; j++) {
				if (same(rates[i], rates[j])) {
					throw new IOException(path + ": merge rate " + rates[i] + " appears twice");
				}
			}
		}
		return new CutoffTable(rates, cutoffs, fallback);
	}

	/** The cutoff for a merge rate: its own row, else the default. */
	public double forMergeRate(double mergeRate) {
		for (int i = 0; i < rates.length; i++) {
			if (same(rates[i], mergeRate)) {
				return cutoffs[i];
			}
		}
		if (fallback == null) {
			throw new IllegalStateException("no cutoff for merge rate " + mergeRate + " and no default cutoff given");
		}
		return fallback;
	}

	private static boolean same(double a, double b) {
		return Math.abs(a - b) <= RATE_TOLERANCE * Math.max(1.0, Math.abs(b));
	}

	/** A number, or inf / -inf. */
	static double parseNumber(String s) {
		s = s.trim().toLowerCase();
		if (s.equals("inf") || s.equals("+inf")) {
			return Double.POSITIVE_INFINITY;
		}
		if (s.equals("-inf")) {
			return Double.NEGATIVE_INFINITY;
		}
		return Double.parseDouble(s);
	}
}
