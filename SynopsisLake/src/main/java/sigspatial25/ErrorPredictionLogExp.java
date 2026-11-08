package sigspatial25;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.TreeSet;

import org.apache.commons.math3.distribution.BetaDistribution;

import histogram.SimpleSpatialHistogramOpt;
import predictor.ApproxAnswer;
import predictor.CutoffTable;
import predictor.ErrorPredictor;
import predictor.GatedRangeQuery;
import predictor.LearnedErrorPredictor;
import predictor.PointScanCounter;
import predictor.PredictorFactory;
import predictor.QueryErrorEstimator;
import predictor.RawRangeCounter;
import predictor.SynopsisContext;
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
 *
 * -modelPath <model.txt> also scores every query with that LightGBM model (predictor.LearnedErrorPredictor)
 * and adds a learnedErr column: the model's estimate of the relative error, to be compared with predErr.
 *
 * Gate (query path, predictor.GatedRangeQuery): with a cutoff (-cutoffs <cutoffs.csv> per merge rate and/or
 * -cutoff <value>, or gate.cutoff in model.properties with -gate true) every query is also run through the gate and one row per
 * (merge rate, query) goes to gated.csv: the answer returned, whether it came from the synopsis or the raw data,
 * the estimated relative error, and the time spent on each step. The gate uses the model given by -modelPath, or
 * Skewness-Align when none is given (or -gatePredictor skew). -rawFallback true reads the raw points (the
 * numFile data files) for queries sent to the raw data; false only records the decision. -eps and -delta pick
 * rows when cutoffs.csv has several.
 */
public class ErrorPredictionLogExp extends CountHistExp {

	String outDir;
	boolean reproduceAuthors;
	boolean fixStride;
	ErrorPredictor learned;
	ErrorPredictor gatePredictor;
	CutoffTable cutoffs;
	RawRangeCounter rawCounter;

	public ErrorPredictionLogExp(String dataDir, int numFile, int[] srcRes, String queryFile, String outDir,
			int end, int step, boolean fixStride, boolean reproduceAuthors, String modelPath) throws IOException {
		this(dataDir, numFile, srcRes, queryFile, outDir, end, step, fixStride, reproduceAuthors, modelPath, null);
	}

	public ErrorPredictionLogExp(String dataDir, int numFile, int[] srcRes, String queryFile, String outDir,
			int end, int step, boolean fixStride, boolean reproduceAuthors, String modelPath, GateOptions gate)
			throws IOException {
		super(dataDir.endsWith("/") ? dataDir : dataDir + "/", "", numFile, 2, 0.5, queryFile);
		setSrcResolution(srcRes);
		this.outDir = outDir.endsWith("/") ? outDir : outDir + "/";
		this.fixStride = fixStride;
		this.reproduceAuthors = reproduceAuthors;
		this.learned = modelPath == null ? null : LearnedErrorPredictor.load(modelPath);
		if (gate != null) {
			this.gatePredictor = gate.skewPredictor ? PredictorFactory.predictor(null, null)
					: (learned != null ? learned : PredictorFactory.predictor(null, null));
			this.cutoffs = PredictorFactory.cutoffs(gate.cutoffsPath, gate.cutoff, gate.eps, gate.delta,
					this.gatePredictor);
			if (this.cutoffs == null) {
				throw new IllegalArgumentException(
						"gate needs -cutoffs, -cutoff, or gate.cutoff in the model's properties");
			}
			if (gate.rawFallback) {
				ArrayList<String> files = new ArrayList<String>();
				for (int f = 0; f < numFile; f++) {
					files.add(this.dataDirWithSlash(dataDir) + f + ".csv");
				}
				this.rawCounter = PointScanCounter.fromFiles(files);
			}
		}
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
				+ "cellsTouched,srcSumEst" + (learned == null ? "" : ",learnedErr") + "\n");

		FileWriter gated = null;
		if (cutoffs != null) {
			gated = new FileWriter(this.outDir + "gated.csv");
			gated.write("method,mergeRate,qId,gt,answer,fromSynopsis,errorScore,cutoff,estRelErr,relErr,"
					+ "gateNs,synNs,rawNs\n");
		}
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
			if (gated != null) {
				logGated(gated, rangeQueries, srcHists, arm);
			}
			System.out.println(String.format("mergeRate %.1f%%, lonRes %d: mean relErr %.3f", mergeRate * 100,
					lonRes, meanRelErr));
		}
		rows.close();
		if (gated != null) {
			gated.close();
		}
	}

	private String dataDirWithSlash(String dataDir) {
		return dataDir.endsWith("/") ? dataDir : dataDir + "/";
	}

	/** Gate options for the query path; null fields mean not given. */
	public static class GateOptions {
		public String cutoffsPath;
		public Double cutoff;
		public Double eps;
		public Double delta;
		public boolean skewPredictor;
		public boolean rawFallback;
	}

	private void logGated(FileWriter out, ArrayList<double[]> rangeQueries, SimpleSpatialHistogramOpt[] srcHists,
			SimpleSpatialHistogramOpt arm) throws IOException {
		GatedRangeQuery gate = new GatedRangeQuery(
				new SynopsisContext(arm, srcHists, mergeRate, beta(arm)),
				gatePredictor, cutoffs.forMergeRate(mergeRate), rawCounter);
		StringBuilder sb = new StringBuilder();
		for (int qId = 0; qId < rangeQueries.size(); qId++) {
			double[] q = rangeQueries.get(qId);
			double gt = q[4];
			ApproxAnswer a = gate.answer(q[0], q[1], q[2], q[3]);
			sb.setLength(0);
			sb.append("ARM,").append(mergeRate).append(',').append(qId).append(',').append(gt).append(',')
					.append(a.answer).append(',').append(a.fromSynopsis).append(',').append(a.errorScore).append(',')
					.append(a.cutoff).append(',').append(a.estimatedRelError).append(',')
					.append(Math.abs(gt - a.answer) / gt).append(',').append(a.gateNanos).append(',')
					.append(a.synopsisNanos).append(',').append(a.rawNanos).append('\n');
			out.write(sb.toString());
		}
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
		QueryErrorEstimator learnedErr = learned == null ? null
				: learned.bind(new SynopsisContext(arm, srcHists, mergeRate, betaArm));
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
					.append(',').append(srcSum);
			if (learnedErr != null) {
				sb.append(',').append(learnedErr.estimate(q[0], q[1], q[2], q[3]));
			}
			sb.append('\n');
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

	/** An unset job parameter arrives as an empty string. */
	private static String emptyToNull(String s) {
		return s.isEmpty() ? null : s;
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
		String modelPath = null;
		GateOptions gate = new GateOptions();
		boolean gateWanted = false;

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
			} else if (args[i].equals("-modelPath")) {
				modelPath = emptyToNull(args[++i]);
			} else if (args[i].equals("-gate")) {
				gateWanted = Boolean.parseBoolean(args[++i]);
			} else if (args[i].equals("-cutoffs")) {
				gate.cutoffsPath = emptyToNull(args[++i]);
				gateWanted = gate.cutoffsPath != null;
			} else if (args[i].equals("-cutoff")) {
				gate.cutoff = Double.valueOf(args[++i]);
				gateWanted = true;
			} else if (args[i].equals("-eps")) {
				gate.eps = Double.valueOf(args[++i]);
			} else if (args[i].equals("-delta")) {
				gate.delta = Double.valueOf(args[++i]);
			} else if (args[i].equals("-gatePredictor")) {
				String g = args[++i];
				if (!g.equals("skew") && !g.equals("learned")) {
					throw new IllegalArgumentException("-gatePredictor must be skew or learned, got " + g);
				}
				gate.skewPredictor = g.equals("skew");
			} else if (args[i].equals("-rawFallback")) {
				gate.rawFallback = Boolean.parseBoolean(args[++i]);
			} else {
				throw new IllegalArgumentException("unknown argument " + args[i]);
			}
		}
		if (srcRes.length != numFile) {
			throw new IllegalArgumentException("-srcRes needs " + numFile + " values, got " + srcRes.length);
		}
		System.out.println("dataDir=" + dataDir + " numFile=" + numFile + " queryFile=" + queryFile + " outDir="
				+ outDir + " end=" + end + " step=" + step + " fixStride=" + fixStride + " reproduceAuthors="
				+ reproduceAuthors + " modelPath=" + modelPath);
		new ErrorPredictionLogExp(dataDir, numFile, srcRes, queryFile, outDir, end, step, fixStride,
				reproduceAuthors, modelPath, gateWanted || gate.rawFallback ? gate : null);
	}
}
