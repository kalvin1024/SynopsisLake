package sigspatial25;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import predictor.QueryFile;
import predictor.SourceShapePredictor;
import predictor.SynopsisStore;

/**
 * Diagnostic export for independent Python/Java reference parity checks.
 * Query labels, if present in the fifth input column, are never read here.
 * This command does not run a gate, train/calibrate, or claim E2E latency.
 */
public final class SourceShapeReferenceExp {
    private SourceShapeReferenceExp() { }

    public static void main(String[] args) throws IOException {
        String shapeDir = null, packet = null, queriesPath = null, output = null, runDir = null;
        Double requestedRate = null;
        for (int i = 0; i < args.length; ++i) {
            String name = args[i];
            if (++i >= args.length) throw new IllegalArgumentException("Missing value for " + name);
            String value = args[i];
            switch (name) {
            case "-shapeDir": shapeDir = value; break;
            case "-shapePacket": packet = value; break;
            case "-queryFile": queriesPath = value; break;
            case "-outFile": output = value; break;
            case "-runDir": runDir = value; break;
            case "-mergeRate": requestedRate = Double.valueOf(value); break;
            default: throw new IllegalArgumentException("Unknown argument " + name);
            }
        }
        if (shapeDir == null || queriesPath == null || output == null || (runDir == null) != (requestedRate == null))
            throw new IllegalArgumentException("Need -shapeDir, -queryFile, -outFile; optional -runDir and -mergeRate together");
        long start = System.nanoTime();
        SourceShapePredictor predictor = SourceShapePredictor.load(shapeDir, packet);
        long loaded = System.nanoTime();
        SynopsisStore store = runDir == null ? null : SynopsisStore.load(runDir);
        double rate = store == null ? Double.NaN : store.nearestMergeRate(requestedRate);
        List<double[]> queries = QueryFile.load(queriesPath);
        File parent = new File(output).getAbsoluteFile().getParentFile();
        if (parent != null) parent.mkdirs();
        try (FileWriter writer = new FileWriter(output)) {
            writer.write("qId,reference_count,H,score\n");
            for (int id = 0; id < queries.size(); ++id) {
                double[] q = queries.get(id);
                double reference = predictor.referenceCount(q[0], q[1], q[2], q[3]);
                double answer = store == null ? Double.NaN : store.merged(rate).RangeQuery(q[0], q[1], q[2], q[3]);
                double score = store == null ? Double.NaN : SourceShapePredictor.score(answer, reference);
                writer.write(id + "," + reference + "," + answer + "," + score + "\n");
            }
        }
        System.out.println("Native/packet initialization_ns=" + (loaded - start)
                + " serialized_bytes=" + predictor.serializedBytes() + " selected_cells=" + predictor.selectedCells()
                + " queries=" + queries.size() + "; reference export is diagnostic, not an E2E benchmark");
    }
}
