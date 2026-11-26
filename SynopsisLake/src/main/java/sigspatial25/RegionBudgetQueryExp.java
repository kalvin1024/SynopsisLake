package sigspatial25;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.zip.GZIPInputStream;

import histogram.SimpleSpatialHistogramOpt;
import predictor.CountHistogramPayload;
import predictor.ErrorPredictor;
import predictor.FeatureRegistry;
import predictor.HistogramIO;
import predictor.PointScanCounter;
import predictor.PredictorFactory;
import predictor.QueryErrorEstimator;
import predictor.QueryFeatures;
import predictor.QueryFile;
import predictor.RawRangeCounter;
import predictor.SkewnessBeta;
import predictor.SourceShapePredictor;
import predictor.SynopsisContext;

/**
 * Shared, single-histogram batch engine for the held-out-region storage study.
 * Existing ApproxQueryExp behavior is unchanged. This entry deliberately loads
 * only the artifacts needed by the selected method, rather than every ARM file.
 *
 * -method exact|source_shape|v1|fine_direct|fine_reference|v1_features
 * -queryFile FILE -outFile FILE [-setupOut FILE]
 * -answerHist FILE       compact SLHIST01 H (SS/v1/reference/features only)
 * -sourceDir DIR         src_*.txt needed by full-v1 (no ARM files are read)
 * -modelPath FILE [-propsPath FILE]       saved LightGBM full-v1 regression
 * -shapeDir DIR [-shapePacket FILE]      existing native/child SS payloads
 * -finePath FILE         existing SLCNT001 exact-bin count payload
 * -mergeRate R -cutoff VALUE -scoreOnly true|false
 * -gatePath FILE         optional 24-byte LE cutoff/epsilon/delta artifact
 * -dataDir DIR -numFile N                numbered raw .csv or .csv.gz files
 * -featureOut FILE       v1_features output; no model or raw points needed
 *
 * Score-only passes expose candidate answers and scores, without loading or
 * scanning raw data. The fifth query column is output-only ground truth; no
 * acceptance rule or feature reads it. Cutoffs are supplied scalars already
 * selected outside this engine. Positive-infinite scores are always rejected.
 */
public final class RegionBudgetQueryExp {
    private RegionBudgetQueryExp() { }

    public static void main(String[] args) throws IOException {
        Map<String, String> options = parse(args);
        String method = required(options, "-method");
        if (!Arrays.asList("exact", "source_shape", "v1", "fine_direct", "fine_reference", "v1_features").contains(method))
            throw new IllegalArgumentException("Unknown -method " + method);
        String queryFile = required(options, "-queryFile"), outFile = required(options, "-outFile");
        boolean featureMode = method.equals("v1_features");
        boolean scoreOnly = featureMode || bool(options.getOrDefault("-scoreOnly", "false"));
        if (method.equals("exact") && scoreOnly) throw new IllegalArgumentException("exact has no score-only mode");
        double cutoff = number(options.getOrDefault("-cutoff", "-inf"));
        if (Double.isNaN(cutoff)) throw new IllegalArgumentException("cutoff must not be NaN");
        double rate = number(options.getOrDefault("-mergeRate", "1"));
        if (!Double.isFinite(rate) || rate < 0) throw new IllegalArgumentException("Invalid -mergeRate");
        // These fields describe the external calibration policy; no local label-based fitting occurs.
        double eps = number(options.getOrDefault("-eps", "0.25"));
        double delta = number(options.getOrDefault("-delta", "0.05"));
        if (!Double.isFinite(eps) || eps <= 0 || !Double.isFinite(delta) || delta < 0 || delta >= 1)
            throw new IllegalArgumentException("Invalid -eps/-delta");

        List<String> stages = new ArrayList<String>();
        long start;
        if (options.containsKey("-gatePath")) {
            if (options.containsKey("-cutoff")) throw new IllegalArgumentException("Use -gatePath or -cutoff, not both");
            start = System.nanoTime();
            byte[] gate = Files.readAllBytes(Paths.get(options.get("-gatePath")));
            if (gate.length != 24) throw new IOException("gate artifact must contain exactly three LE float64 values");
            ByteBuffer values = ByteBuffer.wrap(gate).order(ByteOrder.LITTLE_ENDIAN);
            cutoff = values.getDouble(); eps = values.getDouble(); delta = values.getDouble();
            if (Double.isNaN(cutoff) || !Double.isFinite(eps) || eps <= 0
                    || !Double.isFinite(delta) || delta < 0 || delta >= 1)
                throw new IOException("Invalid cutoff/epsilon/delta gate artifact");
            if ((options.containsKey("-eps") && number(options.get("-eps")) != eps)
                    || (options.containsKey("-delta") && number(options.get("-delta")) != delta))
                throw new IOException("Gate artifact epsilon/delta differs from declared experiment");
            stages.add(stage("gate_artifact_load", start));
        }
        CountHistogramPayload answerHist = null, fine = null;
        SimpleSpatialHistogramOpt[] sources = new SimpleSpatialHistogramOpt[0];
        ErrorPredictor predictor = null;
        QueryErrorEstimator estimator = null;
        SynopsisContext context = null;
        List<String> featureNames = null;
        FeatureRegistry.Feature[] features = null;

        if (Arrays.asList("source_shape", "v1", "fine_reference", "v1_features").contains(method)) {
            start = System.nanoTime();
            answerHist = CountHistogramPayload.load(required(options, "-answerHist"));
            stages.add(stage("answer_histogram_load", start));
            if (!options.containsKey("-mergeRate")) rate = answerHist.mergeRate();
            else if (Math.abs(rate - answerHist.mergeRate()) > 1e-12)
                throw new IOException("Declared merge rate does not match the saved candidate histogram");
        }
        if (method.equals("v1") || featureMode) {
            start = System.nanoTime();
            sources = loadSources(required(options, "-sourceDir"));
            stages.add(stage("source_histograms_load", start));
        }
        if (method.equals("source_shape")) {
            start = System.nanoTime();
            predictor = SourceShapePredictor.load(required(options, "-shapeDir"), options.get("-shapePacket"));
            stages.add(stage("source_shape_load", start));
        } else if (method.equals("v1")) {
            start = System.nanoTime();
            predictor = PredictorFactory.predictor(required(options, "-modelPath"), options.get("-propsPath"));
            stages.add(stage("model_load", start));
        } else if (method.equals("fine_direct") || method.equals("fine_reference")) {
            start = System.nanoTime();
            fine = CountHistogramPayload.load(required(options, "-finePath"));
            // Reject SLHIST01 here: only the exact-bin compact fine codec has count bounds.
            fine.countInterval(fine.histogram().getLonBoundary()[0], fine.histogram().getLonBoundary()[0],
                    fine.histogram().getLatBoundary()[0], fine.histogram().getLatBoundary()[0]);
            stages.add(stage("fine_histogram_load_and_prefix", start));
        }
        if (predictor != null || featureMode) {
            start = System.nanoTime();
            boolean beta = featureMode || predictor.needsBeta();
            context = new SynopsisContext(answerHist.histogram(), sources, rate,
                    beta ? SkewnessBeta.of(answerHist.histogram()) : null);
            if (featureMode) {
                featureNames = FeatureRegistry.names();
                features = FeatureRegistry.resolve(featureNames.toArray(new String[0]));
            } else estimator = predictor.bind(context);
            stages.add(stage("synopsis_context_and_predictor_bind", start));
        }

        RawRangeCounter raw = null;
        if (!scoreOnly) {
            start = System.nanoTime();
            String dataDir = required(options, "-dataDir");
            int numFiles = Integer.parseInt(options.getOrDefault("-numFile", "4"));
            if (numFiles < 1) throw new IllegalArgumentException("-numFile must be positive");
            raw = loadRaw(dataDir, numFiles);
            stages.add(stage("raw_points_load", start));
        }

        start = System.nanoTime();
        List<double[]> queries = QueryFile.load(queryFile);
        for (double[] q : queries) validateQuery(q);
        stages.add(stage("query_file_load", start));
        ensureParent(outFile);
        FileWriter featureOut = null;
        if (featureMode) {
            String path = required(options, "-featureOut");
            ensureParent(path);
            featureOut = new FileWriter(path);
            featureOut.write("qId,mergeRate,candidate,minLon,maxLon,minLat,maxLat");
            for (String name : featureNames) featureOut.write("," + name);
            featureOut.write("\n");
        }
        int accepted = 0, fallbacks = 0;
        start = System.nanoTime();
        try (FileWriter out = new FileWriter(outFile); FileWriter f = featureOut) {
            out.write("qId,mergeRate,candidate,answer,fromSynopsis,estRelError,errorScore,cutoff,gateNs,synNs,rawNs,exact,relErr,countLower,countUpper\n");
            for (int qId = 0; qId < queries.size(); qId++) {
                double[] q = queries.get(qId);
                long gateNs = 0, synNs = 0, rawNs = 0;
                double candidate = Double.NaN, score = Double.NaN, lower = Double.NaN, upper = Double.NaN;
                long t;
                if (!method.equals("exact")) {
                    t = System.nanoTime();
                    candidate = method.equals("fine_direct") ? fine.count(q[0], q[1], q[2], q[3])
                            : answerHist.count(q[0], q[1], q[2], q[3]);
                    synNs = System.nanoTime() - t;
                    t = System.nanoTime();
                    if (method.equals("fine_direct")) {
                        double[] interval = fine.countInterval(q[0], q[1], q[2], q[3]);
                        lower = interval[0]; upper = interval[1];
                        score = CountHistogramPayload.relativeErrorBound(candidate, lower, upper);
                    } else if (method.equals("fine_reference")) {
                        double reference = fine.count(q[0], q[1], q[2], q[3]);
                        score = SourceShapePredictor.score(candidate, reference);
                    } else if (featureMode) {
                        QueryFeatures values = new QueryFeatures(context, q[0], q[1], q[2], q[3]);
                        StringBuilder row = new StringBuilder().append(qId).append(',').append(rate)
                                .append(',').append(candidate);
                        for (int i = 0; i < 4; i++) row.append(',').append(q[i]);
                        for (FeatureRegistry.Feature feature : features) row.append(',').append(feature.value(values));
                        f.write(row.append('\n').toString());
                    } else score = estimator.estimate(q[0], q[1], q[2], q[3]);
                    gateNs = System.nanoTime() - t;
                }
                boolean fromSynopsis = !method.equals("exact") && !featureMode
                        && Double.isFinite(score) && score <= cutoff;
                double answer = candidate;
                if (fromSynopsis) accepted++;
                else if (!scoreOnly) {
                    t = System.nanoTime();
                    answer = raw.count(q[0], q[1], q[2], q[3]);
                    rawNs = System.nanoTime() - t;
                    fallbacks++;
                }
                double estimated = method.equals("exact") || (!fromSynopsis && !scoreOnly) ? 0 : score;
                StringBuilder row = new StringBuilder().append(qId).append(',').append(rate).append(',')
                        .append(candidate).append(',').append(answer).append(',').append(fromSynopsis).append(',')
                        .append(estimated).append(',').append(score).append(',')
                        .append(method.equals("exact") ? Double.NaN : cutoff).append(',')
                        .append(gateNs).append(',').append(synNs).append(',').append(rawNs).append(',');
                if (Double.isNaN(q[4])) row.append(',');
                else row.append(q[4]).append(',').append(relativeError(answer, q[4]));
                row.append(',').append(lower).append(',').append(upper).append('\n');
                out.write(row.toString());
            }
        }
        stages.add(stage("answer_and_csv_output_loop", start));
        writeStages(options.get("-setupOut"), stages);
        System.out.println(String.format(Locale.ROOT,
                "%s: %d queries, %d accepted, %d actual raw scans, scoreOnly=%s; wrote %s",
                method, queries.size(), accepted, fallbacks, scoreOnly, outFile));
    }

    private static SimpleSpatialHistogramOpt[] loadSources(String dir) throws IOException {
        File[] files = new File(dir).listFiles((d, n) -> n.matches("src_\\d+\\.txt"));
        if (files == null || files.length == 0) throw new IOException(dir + ": no src_<id>.txt source histograms");
        Arrays.sort(files, Comparator.comparingInt(f -> Integer.parseInt(f.getName().replaceAll("\\D", ""))));
        SimpleSpatialHistogramOpt[] result = new SimpleSpatialHistogramOpt[files.length];
        for (int i = 0; i < files.length; i++) result[i] = HistogramIO.read(files[i].getPath()).hist;
        return result;
    }

    private static Map<String, String> parse(String[] args) {
        List<String> keys = Arrays.asList("-method", "-queryFile", "-outFile", "-setupOut", "-answerHist",
                "-sourceDir", "-modelPath", "-propsPath", "-shapeDir", "-shapePacket", "-finePath",
                "-mergeRate", "-cutoff", "-gatePath", "-scoreOnly", "-dataDir", "-numFile", "-featureOut", "-eps", "-delta");
        Map<String, String> options = new LinkedHashMap<String, String>();
        for (int i = 0; i < args.length; i++) {
            String key = args[i];
            if (!keys.contains(key)) throw new IllegalArgumentException("Unknown argument " + key);
            if (++i == args.length || args[i].isEmpty()) throw new IllegalArgumentException("Missing value for " + key);
            if (options.put(key, args[i]) != null) throw new IllegalArgumentException("Duplicate argument " + key);
        }
        return options;
    }

    private static String required(Map<String, String> options, String key) {
        String value = options.get(key);
        if (value == null || value.isEmpty()) throw new IllegalArgumentException("Needs " + key);
        return value;
    }

    /** Original point-scan implementation, with format-compatible streaming
     * loading for the local compressed AUTHORS files. Scans still visit the
     * actual loaded points; this loader never reads query truth or cached counts.
     */
    private static PointScanCounter loadRaw(String dir, int count) throws IOException {
        PointScanCounter counter = new PointScanCounter();
        for (int id = 0; id < count; id++) {
            Path path = Paths.get(dir, id + ".csv");
            boolean gzip = false;
            if (!Files.isRegularFile(path)) {
                path = Paths.get(dir, id + ".csv.gz");
                gzip = true;
            }
            if (!Files.isRegularFile(path)) throw new IOException("Missing raw source " + id + " in " + dir);
            InputStream stream = Files.newInputStream(path);
            if (gzip) stream = new GZIPInputStream(stream);
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
                String line;
                long lineNo = 0;
                while ((line = reader.readLine()) != null) {
                    lineNo++;
                    String[] fields = line.trim().split("\\s+");
                    if (fields.length < 2 || fields[0].isEmpty())
                        throw new IOException(path + ":" + lineNo + ": needs unweighted x/y fields");
                    double x, y;
                    try { x = Double.parseDouble(fields[0]); y = Double.parseDouble(fields[1]); }
                    catch (NumberFormatException e) { throw new IOException(path + ":" + lineNo + ": invalid point", e); }
                    if (!Double.isFinite(x) || !Double.isFinite(y))
                        throw new IOException(path + ":" + lineNo + ": nonfinite point");
                    counter.add(x, y);
                }
            }
        }
        return counter;
    }

    private static boolean bool(String value) {
        if (!value.equals("true") && !value.equals("false")) throw new IllegalArgumentException("Boolean must be true or false");
        return Boolean.parseBoolean(value);
    }

    private static double number(String value) {
        String lower = value.toLowerCase(Locale.ROOT);
        if (lower.equals("inf") || lower.equals("+inf") || lower.equals("infinity") || lower.equals("+infinity"))
            return Double.POSITIVE_INFINITY;
        if (lower.equals("-inf") || lower.equals("-infinity")) return Double.NEGATIVE_INFINITY;
        return Double.parseDouble(value);
    }

    private static String stage(String name, long start) { return name + "," + (System.nanoTime() - start); }
    private static void ensureParent(String path) throws IOException {
        File parent = new File(path).getAbsoluteFile().getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) throw new IOException("Cannot create " + parent);
    }
    private static void writeStages(String path, List<String> rows) throws IOException {
        if (path == null) return;
        ensureParent(path);
        try (FileWriter out = new FileWriter(path)) {
            out.write("stage,elapsed_ns\n");
            for (String row : rows) out.write(row + "\n");
        }
    }
    private static void validateQuery(double[] q) {
        for (int i = 0; i < 4; i++) if (!Double.isFinite(q[i])) throw new IllegalArgumentException("Nonfinite query bounds");
        if (q[0] > q[1] || q[2] > q[3]) throw new IllegalArgumentException("Reversed query bounds");
        if (!Double.isNaN(q[4]) && (!Double.isFinite(q[4]) || q[4] < 0))
            throw new IllegalArgumentException("Ground-truth output label must be a nonnegative finite count");
    }
    static double relativeError(double answer, double truth) {
        return truth > 0 ? Math.abs(answer - truth) / truth : answer == 0 ? 0 : Double.POSITIVE_INFINITY;
    }
}
