package sigspatial25;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.TreeSet;

import org.apache.commons.math3.distribution.BetaDistribution;

import histogram.SimpleSpatialHistogramOpt;
import utils.UtilsFunction;

/**
 * Per-query logger for the ML error predictor (formats: ml/SCHEMA.md).
 *
 * Builds the data-driven ARM histogram the same way as
 * DataDrivenRangeQueryMergingRateExp.mergedPrecisionExp, for every merge rate from the canonical
 * resolution down to -end in steps of -step, and writes one row per (merge rate, query) to rows.csv
 * plus the merged (arm_<lonRes>.txt) and source (src_<i>.txt) histograms.
 *
 * -reproduceAuthors true reloads the sources at every merge rate like the authors' driver does
 * (each reload widens the global bounds by varepsilon again), which reproduces
 * dataDrivenQuery/results/DataDrivenQueryAvgRelError.txt exactly. -fixStride true computes the beta
 * distribution with the correct row stride (data[0].length instead of data.length).
 */
public class ErrorPredictionLogExp extends CountHistExp {

	String outDir;
	boolean reproduceAuthors;
	boolean fixStride;

	public ErrorPredictionLogExp(String dataDir, int numFile, int[] srcRes, String queryFile, String outDir,
			int end, int step, boolean fixStride, boolean reproduceAuthors) throws IOException {
		super(dataDir.endsWith("/") ? dataDir : dataDir + "/", "", numFile, 2, 0.5, queryFile);
		setSrcResolution(srcRes);
		this.outDir = outDir.endsWith("/") ? outDir : outDir + "/";
		this.fixStride = fixStride;
		this.reproduceAuthors = reproduceAuthors;
		new File(this.outDir).mkdirs();

		SimpleSpatialHistogramOpt[] srcHists = loadSynopsisFromData();
		for (int sId = 0; sId < srcHists.length; sId++) {
			writeHist(this.outDir + "src_" + sId + ".txt", srcHists[sId], 1.0);
		}
		int canonicalLon = numDistinct(srcHists, true) - 1;
		int canonicalLat = numDistinct(srcHists, false) - 1;
		System.out.println("canonical resolution = " + canonicalLon + " x " + canonicalLat);

		ArrayList<double[]> rangeQueries = loadRangeQuery(queryFile);
		FileWriter rows = new FileWriter(this.outDir + "rows.csv");
		rows.write("method,mergeRate,lonRes,latRes,qId,gt,est,relErr,predErr,betaAlpha,overlapSum,"
				+ "cellsTouched,srcSumEst\n");

		boolean first = true;
		for (int lonRes = canonicalLon; lonRes >= end; lonRes -= step) {
			lonTargetResolution = lonRes;
			latTargetResolution = lonRes;
			mergeRate = lonRes * lonRes / (double) canonicalLon / (double) canonicalLat;
			if (first) {
				latTargetResolution = canonicalLat;
				mergeRate = 1.0;
			} else if (reproduceAuthors) {
				srcHists = loadSynopsisFromData();
			}
			first = false;

			SimpleSpatialHistogramOpt arm = buildArm(srcHists);
			writeHist(this.outDir + "arm_" + lonRes + ".txt", arm, mergeRate);
			double meanRelErr = logQueries(rows, rangeQueries, srcHists, arm);
			System.out.println(String.format("mergeRate %.1f%%, lonRes %d: mean relErr %.3f", mergeRate * 100,
					lonRes, meanRelErr));
		}
		rows.close();
	}

	/** Same steps as DataDrivenRangeQueryMergingRateExp.mergedPrecisionExp for alignCombineHist. */
	private SimpleSpatialHistogramOpt buildArm(SimpleSpatialHistogramOpt[] srcHists) {
		ArrayList<double[]> alignCombineSolutions = dataDrivenAlignSolution(srcHists);
		SimpleSpatialHistogramOpt alignCombineHist = new SimpleSpatialHistogramOpt(MinMinLon, MinMinLat, MaxMaxLon,
				MaxMaxLat, alignCombineSolutions.get(0).length - 1, alignCombineSolutions.get(1).length - 1,
				alignCombineSolutions.get(0), alignCombineSolutions.get(1));
		for (int sId = 0; sId < srcHists.length; sId++) {
			alignCombineHist.aggregateHistogramSep(srcHists[sId]);
		}
		return alignCombineHist;
	}

	private double logQueries(FileWriter rows, ArrayList<double[]> rangeQueries, SimpleSpatialHistogramOpt[] srcHists,
			SimpleSpatialHistogramOpt arm) throws IOException {
		BetaDistribution betaArm = beta(arm);
		BetaDistribution[] betaSrc = new BetaDistribution[srcHists.length];
		for (int sId = 0; sId < srcHists.length; sId++) {
			betaSrc[sId] = beta(srcHists[sId]);
		}
		int lonRes = arm.getLonBoundary().length - 1;
		int latRes = arm.getLatBoundary().length - 1;
		double totalRelErr = 0;
		StringBuilder sb = new StringBuilder();
		for (int qId = 0; qId < rangeQueries.size(); qId++) {
			double[] q = rangeQueries.get(qId);
			double gt = q[4];
			// answers: 0 estimate, 2 Skewness-Align error ratio, 3 sum of overlap ratios, 4 cells touched
			// (length 3 when the query misses the histogram)
			double[] ans = arm.RangeQueryWithRatio(betaArm, q[0], q[1], q[2], q[3]);
			double srcSum = 0;
			for (int sId = 0; sId < srcHists.length; sId++) {
				srcSum += srcHists[sId].RangeQueryWithRatio(betaSrc[sId], q[0], q[1], q[2], q[3])[0];
			}
			double relErr = Math.abs(gt - ans[0]) / gt;
			totalRelErr += relErr;
			sb.setLength(0);
			sb.append("ARM,").append(mergeRate).append(',').append(lonRes).append(',').append(latRes).append(',')
					.append(qId).append(',').append(gt).append(',').append(ans[0]).append(',').append(relErr)
					.append(',').append(ans[2]).append(',').append(betaArm.getAlpha()).append(',')
					.append(ans.length > 3 ? ans[3] : 0.0).append(',').append(ans.length > 4 ? (int) ans[4] : 0)
					.append(',').append(srcSum).append('\n');
			rows.write(sb.toString());
		}
		return totalRelErr / rangeQueries.size();
	}

	private BetaDistribution beta(SimpleSpatialHistogramOpt hist) {
		if (!fixStride) {
			return computeBeta(hist);
		}
		// UtilsFunctionHistogram.ComputeBetaDist with densities[j * data[0].length + k]
		double[] lon = hist.getLonBoundary();
		double[] lat = hist.getLatBoundary();
		double[][] data = hist.getData();
		double[] densities = new double[data.length * data[0].length];
		for (int j = 0; j < lon.length - 1; j++) {
			for (int k = 0; k < lat.length - 1; k++) {
				densities[j * data[0].length + k] = data[j][k] / (lon[j + 1] - lon[j]) / (lat[k + 1] - lat[k]);
			}
		}
		double median = UtilsFunction.CalculateMedian(densities);
		double mad = UtilsFunction.computeMADByLibrary(densities);
		double dk = mad / (median + varepsilon);
		double param = scaleFactor / (dk + varepsilon) + 1;
		return new BetaDistribution(param, param);
	}

	private static int numDistinct(SimpleSpatialHistogramOpt[] hists, boolean lon) {
		TreeSet<Double> set = new TreeSet<Double>();
		for (SimpleSpatialHistogramOpt h : hists) {
			for (double b : lon ? h.getLonBoundary() : h.getLatBoundary()) {
				set.add(b);
			}
		}
		return set.size();
	}

	/** Line 1 merge rate, line 2 lon boundaries, line 3 lat boundaries, then one row of counts per lon bucket. */
	private static void writeHist(String path, SimpleSpatialHistogramOpt hist, double rate) throws IOException {
		FileWriter w = new FileWriter(path);
		w.write(rate + "\n");
		w.write(join(hist.getLonBoundary()) + "\n");
		w.write(join(hist.getLatBoundary()) + "\n");
		for (double[] row : hist.getData()) {
			w.write(join(row) + "\n");
		}
		w.close();
	}

	private static String join(double[] values) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < values.length; i++) {
			if (i > 0) {
				sb.append(',');
			}
			sb.append(values[i]);
		}
		return sb.toString();
	}

	private static int[] parseInts(String csv) {
		String[] parts = csv.split(",");
		int[] out = new int[parts.length];
		for (int i = 0; i < parts.length; i++) {
			out[i] = Integer.parseInt(parts[i].trim());
		}
		return out;
	}

	public static void main(String[] args) throws IOException {
		String dataDir = "../range_query/";
		int numFile = 4;
		int[] srcRes = { 40, 56, 32, 48 };
		String queryFile = "../range_query/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv";
		String outDir = "../out/ml/authors_range/";
		int end = 42;
		int step = 8;
		boolean fixStride = false;
		boolean reproduceAuthors = false;

		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-dataDir")) {
				dataDir = args[++i];
			} else if (args[i].equals("-numFile")) {
				numFile = Integer.parseInt(args[++i]);
			} else if (args[i].equals("-srcRes")) {
				srcRes = parseInts(args[++i]);
			} else if (args[i].equals("-queryFile")) {
				queryFile = args[++i];
			} else if (args[i].equals("-outDir")) {
				outDir = args[++i];
			} else if (args[i].equals("-end")) {
				end = Integer.parseInt(args[++i]);
			} else if (args[i].equals("-step")) {
				step = Integer.parseInt(args[++i]);
			} else if (args[i].equals("-fixStride")) {
				fixStride = Boolean.parseBoolean(args[++i]);
			} else if (args[i].equals("-reproduceAuthors")) {
				reproduceAuthors = Boolean.parseBoolean(args[++i]);
			} else {
				throw new IllegalArgumentException("unknown argument " + args[i]);
			}
		}
		if (srcRes.length != numFile) {
			throw new IllegalArgumentException("-srcRes needs " + numFile + " values, got " + srcRes.length);
		}
		System.out.println("dataDir=" + dataDir + " numFile=" + numFile + " queryFile=" + queryFile + " outDir="
				+ outDir + " end=" + end + " step=" + step + " fixStride=" + fixStride + " reproduceAuthors="
				+ reproduceAuthors);
		new ErrorPredictionLogExp(dataDir, numFile, srcRes, queryFile, outDir, end, step, fixStride,
				reproduceAuthors);
	}
}
