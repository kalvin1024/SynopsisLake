package sigspatial25;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import predictor.ApproxAnswer;
import predictor.CutoffTable;
import predictor.ErrorPredictor;
import predictor.GatedRangeQuery;
import predictor.MergeRatePlanner;
import predictor.PointScanCounter;
import predictor.PredictorFactory;
import predictor.QueryFile;
import predictor.RawRangeCounter;
import predictor.SynopsisStore;

/**
 * Approximate range counts with estimated relative errors, from saved synopses (no raw data needed unless the gate
 * is allowed to fall back to it).
 *
 * Loads the run folder written by ErrorPredictionLogExp (src_*.txt, arm_*.txt), a LightGBM model (or
 * Skewness-Align when none is given) and a cutoff, and answers every query in -queryFile
 * (minLon,maxLon,minLat,maxLat[,exact] per line) with predictor.GatedRangeQuery. Writes -outFile:
 *
 *   qId,mergeRate,answer,fromSynopsis,estRelError,errorScore,cutoff,gateNs,synNs,rawNs[,exact,relErr]
 *
 * Inputs:
 *   -runDir DIR           run folder (required)
 *   -queryFile FILE       queries (required)
 *   -outFile FILE         default approx.csv
 *   -modelPath FILE       model.txt; model.properties next to it is read (-propsPath to name another)
 *   -cutoffs FILE | -cutoff X   gate cutoffs (cutoffs.csv per merge rate, or one value); else gate.cutoff from the
 *                         model's properties. -eps / -delta (default 0.10 / 0.05) pick rows of a cutoffs file.
 *   -mergeRate R          which merged synopsis to answer from (the nearest one is used)
 *   -plan true            instead of -mergeRate, choose the smallest merge rate that meets eps/delta on
 *                         -planQueryFile (queries with exact counts, ideally not the ones the cutoffs were
 *                         calibrated on); the table is written to <outFile>.plan.csv. -minCoverage (0.01) and
 *                         -minExact (10) tune it.
 *   -mode exact|approx|gated   exact: answer every query from the raw data (-dataDir, -numFile, -queryFile only;
 *                         no run folder, model or cutoff needed). approx: synopsis only. gated: synopsis, raw for
 *                         rejected queries (same as -rawFallback true). Default: approx, or gated with -rawFallback.
 *   -rawFallback true     read the raw points (-dataDir with 0..numFile-1.csv, -numFile) for queries the gate
 *                         rejects; otherwise those queries return the synopsis estimate with fromSynopsis=false
 */
public class ApproxQueryExp {

	public static void main(String[] args) throws IOException {
		String runDir = null, queryFile = null, outFile = "approx.csv", modelPath = null, propsPath = null;
		String cutoffsPath = null, planQueryFile = null, dataDir = null;
		Double cutoff = null, mergeRate = null;
		double eps = 0.10, delta = 0.05, minCoverage = 0.01, minExact = 10;
		boolean plan = false, rawFallback = false;
		String mode = null;
		int numFile = 4;
		for (int i = 0; i < args.length; i++) {
			String a = args[i];
			if (i + 1 >= args.length) {
				throw new IllegalArgumentException("missing value for " + a);
			}
			String v = args[++i];
			if (v.isEmpty()) {
				continue; // an unset job parameter
			}
			switch (a) {
			case "-runDir": runDir = v; break;
			case "-queryFile": queryFile = v; break;
			case "-outFile": outFile = v; break;
			case "-modelPath": modelPath = v; break;
			case "-propsPath": propsPath = v; break;
			case "-cutoffs": cutoffsPath = v; break;
			case "-cutoff": cutoff = Double.valueOf(v); break;
			case "-eps": eps = Double.parseDouble(v); break;
			case "-delta": delta = Double.parseDouble(v); break;
			case "-mergeRate": mergeRate = Double.valueOf(v); break;
			case "-plan": plan = Boolean.parseBoolean(v); break;
			case "-planQueryFile": planQueryFile = v; break;
			case "-minCoverage": minCoverage = Double.parseDouble(v); break;
			case "-minExact": minExact = Double.parseDouble(v); break;
			case "-mode": mode = v; break;
			case "-rawFallback": rawFallback = Boolean.parseBoolean(v); break;
			case "-dataDir": dataDir = v; break;
			case "-numFile": numFile = Integer.parseInt(v); break;
			default: throw new IllegalArgumentException("unknown argument " + a);
			}
		}
		if (mode != null && !mode.equals("exact") && !mode.equals("approx") && !mode.equals("gated")) {
			throw new IllegalArgumentException("-mode must be exact, approx or gated");
		}
		if ("gated".equals(mode)) {
			rawFallback = true;
		} else if ("approx".equals(mode)) {
			rawFallback = false;
		}
		if ("exact".equals(mode)) {
			runExact(queryFile, outFile, dataDir, numFile);
			return;
		}
		if (runDir == null || queryFile == null) {
			throw new IllegalArgumentException("-runDir and -queryFile are required");
		}
		if (plan == (mergeRate != null)) {
			throw new IllegalArgumentException("give exactly one of -mergeRate and -plan true");
		}
		if (plan && planQueryFile == null) {
			throw new IllegalArgumentException("-plan true needs -planQueryFile (queries with exact counts)");
		}
		if (rawFallback && dataDir == null) {
			throw new IllegalArgumentException("-rawFallback true needs -dataDir");
		}

		SynopsisStore store = SynopsisStore.load(runDir);
		ErrorPredictor predictor = PredictorFactory.predictor(modelPath, propsPath);
		CutoffTable cutoffs = PredictorFactory.cutoffs(cutoffsPath, cutoff, eps, delta, predictor);
		if (cutoffs == null) {
			throw new IllegalArgumentException("needs -cutoffs, -cutoff, or gate.cutoff in the model's properties");
		}

		double rate;
		if (plan) {
			MergeRatePlanner.Plan p = new MergeRatePlanner(eps, delta, minCoverage, minExact).plan(store, predictor,
					cutoffs, QueryFile.load(planQueryFile));
			writePlan(outFile + ".plan.csv", p);
			if (p.chosenMergeRate == null) {
				System.out.println("no merge rate meets eps=" + eps + " delta=" + delta + " with coverage >= "
						+ minCoverage + "; read the raw data (see " + outFile + ".plan.csv)");
				if (!rawFallback) {
					System.exit(2);
				}
				rate = store.mergeRates().get(0);
				cutoffs = CutoffTable.constant(Double.NEGATIVE_INFINITY);
			} else {
				rate = p.chosenMergeRate;
				System.out.println("chosen merge rate " + rate);
			}
		} else {
			rate = store.nearestMergeRate(mergeRate);
			System.out.println("answering from merge rate " + rate);
		}

		RawRangeCounter raw = null;
		if (rawFallback) {
			List<String> files = new ArrayList<String>();
			String d = dataDir.endsWith("/") ? dataDir : dataDir + "/";
			for (int f = 0; f < numFile; f++) {
				files.add(d + f + ".csv");
			}
			raw = PointScanCounter.fromFiles(files);
		}
		GatedRangeQuery gate = new GatedRangeQuery(store.context(rate, predictor.needsBeta()), predictor,
				cutoffs.forMergeRate(rate), raw);

		File parent = new File(outFile).getAbsoluteFile().getParentFile();
		if (parent != null) {
			parent.mkdirs();
		}
		List<double[]> queries = QueryFile.load(queryFile);
		int fromSynopsis = 0;
		try (FileWriter out = new FileWriter(outFile)) {
			out.write("qId,mergeRate,answer,fromSynopsis,estRelError,errorScore,cutoff,gateNs,synNs,rawNs,exact,relErr\n");
			StringBuilder sb = new StringBuilder();
			for (int qId = 0; qId < queries.size(); qId++) {
				double[] q = queries.get(qId);
				ApproxAnswer r = gate.answer(q[0], q[1], q[2], q[3]);
				fromSynopsis += r.fromSynopsis ? 1 : 0;
				sb.setLength(0);
				sb.append(qId).append(',').append(rate).append(',').append(r.answer).append(',')
						.append(r.fromSynopsis).append(',').append(r.estimatedRelError).append(',')
						.append(r.errorScore).append(',').append(r.cutoff).append(',').append(r.gateNanos)
						.append(',').append(r.synopsisNanos).append(',').append(r.rawNanos).append(',');
				if (Double.isNaN(q[4])) {
					sb.append(",");
				} else {
					sb.append(q[4]).append(',').append(Math.abs(q[4] - r.answer) / q[4]);
				}
				sb.append('\n');
				out.write(sb.toString());
			}
		}
		System.out.println(String.format(Locale.ROOT, "%d queries, %.1f%% answered from the synopsis; wrote %s",
				queries.size(), 100.0 * fromSynopsis / Math.max(1, queries.size()), outFile));
	}

	private static void runExact(String queryFile, String outFile, String dataDir, int numFile) throws IOException {
		if (queryFile == null || dataDir == null) {
			throw new IllegalArgumentException("-mode exact needs -queryFile and -dataDir");
		}
		List<String> files = new ArrayList<String>();
		String d = dataDir.endsWith("/") ? dataDir : dataDir + "/";
		for (int f = 0; f < numFile; f++) {
			files.add(d + f + ".csv");
		}
		GatedRangeQuery gate = GatedRangeQuery.exact(PointScanCounter.fromFiles(files));
		File parent = new File(outFile).getAbsoluteFile().getParentFile();
		if (parent != null) {
			parent.mkdirs();
		}
		List<double[]> queries = QueryFile.load(queryFile);
		try (FileWriter out = new FileWriter(outFile)) {
			out.write("qId,mergeRate,answer,fromSynopsis,estRelError,errorScore,cutoff,gateNs,synNs,rawNs,exact,relErr\n");
			for (int qId = 0; qId < queries.size(); qId++) {
				double[] q = queries.get(qId);
				ApproxAnswer r = gate.answer(q[0], q[1], q[2], q[3]);
				StringBuilder sb = new StringBuilder();
				sb.append(qId).append(",,").append(r.answer).append(",false,0.0,,,0,0,").append(r.rawNanos)
						.append(',');
				if (Double.isNaN(q[4])) {
					sb.append(",");
				} else {
					sb.append(q[4]).append(',').append(q[4] == 0 ? Double.NaN : Math.abs(q[4] - r.answer) / q[4]);
				}
				out.write(sb.append('\n').toString());
			}
		}
		System.out.println(String.format(Locale.ROOT, "%d queries answered exactly from the raw data; wrote %s",
				queries.size(), outFile));
	}

	private static void writePlan(String path, MergeRatePlanner.Plan plan) throws IOException {
		File parent = new File(path).getAbsoluteFile().getParentFile();
		if (parent != null) {
			parent.mkdirs();
		}
		try (FileWriter w = new FileWriter(path)) {
			w.write("mergeRate,cutoff,queries,answered,coverage,violation,meanRelErrAnswered,meetsTarget,chosen\n");
			for (MergeRatePlanner.RateResult r : plan.rates) {
				w.write(r.mergeRate + "," + r.cutoff + "," + r.queries + "," + r.answered + "," + r.coverage + ","
						+ r.violation + "," + r.meanRelErrAnswered + "," + r.meetsTarget + ","
						+ (plan.chosenMergeRate != null && r.mergeRate == plan.chosenMergeRate) + "\n");
			}
		}
	}
}
