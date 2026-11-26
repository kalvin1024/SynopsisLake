package sigspatial25;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPOutputStream;

import org.junit.Test;

public class RegionBudgetQueryExpTest {
    private static Path fine(Path folder) throws Exception {
        ByteBuffer b = ByteBuffer.allocate(80).order(ByteOrder.LITTLE_ENDIAN);
        b.put("SLCNT001".getBytes(StandardCharsets.US_ASCII)).putInt(2).putInt(1);
        b.putDouble(0).putDouble(2).putDouble(0).putDouble(2).position(64);
        for (int v : new int[] {1, 2, 3, 4}) b.putInt(v);
        Path p = folder.resolve("fine.counts"); Files.write(p, b.array()); return p;
    }
    private static Path original(Path folder) throws Exception {
        ByteBuffer b = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
        b.put("SLHIST01".getBytes(StandardCharsets.US_ASCII)).putInt(1).putInt(1).putDouble(.5);
        for (double v : new double[] {0, 2, 0, 2, 10}) b.putDouble(v);
        Path p = folder.resolve("H.hist"); Files.write(p, b.array()); return p;
    }
    private static void raw(Path folder, String points) throws Exception {
        Files.write(folder.resolve("0.csv"), points.getBytes(StandardCharsets.UTF_8));
    }
    private static Path query(Path folder, String name, String rows) throws Exception {
        Path p = folder.resolve(name); Files.write(p, rows.getBytes(StandardCharsets.UTF_8)); return p;
    }
    private static String[] run(Path folder, String tag, String method, String... args) throws Exception {
        Path out = folder.resolve(tag + ".csv");
        List<String> command = new ArrayList<String>(Arrays.asList("-method", method, "-outFile", out.toString(),
                "-setupOut", folder.resolve(tag + "-setup.csv").toString()));
        command.addAll(Arrays.asList(args));
        RegionBudgetQueryExp.main(command.toArray(new String[0]));
        return Files.readAllLines(out).get(1).split(",", -1);
    }
    private static double value(String[] row, int index) { return Double.parseDouble(row[index]); }

    @Test
    public void directHistogramReturnsFineAnswerReferenceControlReturnsOriginalH() throws Exception {
        Path root = Files.createTempDirectory("region-budget-");
        Path fine = fine(root), h = original(root), q = query(root, "queries.csv", "0,1,0,1,999\n");
        String[] direct = run(root, "direct", "fine_direct", "-finePath", fine.toString(), "-queryFile", q.toString(),
                "-cutoff", "Infinity", "-scoreOnly", "true");
        String[] reference = run(root, "reference", "fine_reference", "-finePath", fine.toString(),
                "-answerHist", h.toString(), "-queryFile", q.toString(), "-cutoff", "Infinity", "-scoreOnly", "true");
        assertEquals(1, value(direct, 2), 0);
        assertEquals(1, value(direct, 3), 0);
        assertEquals(2.5, value(reference, 2), 0);
        assertEquals(2.5, value(reference, 3), 0);
        assertEquals(.9, value(direct, 6), 1e-12);
        assertEquals(1.5, value(reference, 6), 1e-12);
        assertEquals("true", direct[4]); assertEquals("true", reference[4]);
        assertEquals(0, value(direct, 10), 0);
    }

    @Test
    public void infiniteFineUncertaintyTriggersRealFallbackEvenWithInfiniteCutoff() throws Exception {
        Path root = Files.createTempDirectory("region-budget-fallback-");
        Path f = fine(root), q = query(root, "queries.csv", ".25,.75,.25,.75,999\n");
        raw(root, ".3\t.3\n1.5\t1.5\n");
        String[] row = run(root, "fallback", "fine_direct", "-finePath", f.toString(), "-queryFile", q.toString(),
                "-cutoff", "inf", "-dataDir", root.toString(), "-numFile", "1");
        assertEquals(.25, value(row, 2), 1e-12);
        assertEquals(1, value(row, 3), 0); // Not the deliberately false stored label 999.
        assertEquals("false", row[4]); assertTrue(value(row, 10) > 0);
        raw(root, ".3\t.3\n.4\t.4\n1.5\t1.5\n");
        String[] changed = run(root, "fallback_changed", "fine_direct", "-finePath", f.toString(), "-queryFile", q.toString(),
                "-cutoff", "inf", "-dataDir", root.toString(), "-numFile", "1");
        assertEquals(2, value(changed, 3), 0);
    }

    @Test
    public void binaryGateIsReadAndGzipSourcesUseTheSameExactPointScanner() throws Exception {
        Path root = Files.createTempDirectory("region-budget-gzip-");
        Path f = fine(root), q = query(root, "queries.csv", "0,1,0,1\n");
        ByteBuffer gate = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
        gate.putDouble(Double.NEGATIVE_INFINITY).putDouble(.25).putDouble(.05);
        Path g = root.resolve("gate.bin"); Files.write(g, gate.array());
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (GZIPOutputStream out = new GZIPOutputStream(bytes)) {
            out.write(".3 .3 ignored\n1 1 boundary\n1.5 1.5 outside\n".getBytes(StandardCharsets.UTF_8));
        }
        Files.write(root.resolve("0.csv.gz"), bytes.toByteArray());
        String[] row = run(root, "gate", "fine_direct", "-finePath", f.toString(), "-queryFile", q.toString(),
                "-gatePath", g.toString(), "-dataDir", root.toString(), "-numFile", "1");
        assertEquals("false", row[4]); assertEquals(2, value(row, 3), 0);
        String stages = new String(Files.readAllBytes(root.resolve("gate-setup.csv")), StandardCharsets.UTF_8);
        assertTrue(stages.contains("gate_artifact_load,")); assertTrue(stages.contains("raw_points_load,"));
        String[] exact = run(root, "exact", "exact", "-queryFile", q.toString(), "-dataDir", root.toString(), "-numFile", "1");
        assertEquals(2, value(exact, 3), 0);
        assertEquals("false", exact[4]); assertTrue(value(exact, 10) > 0);
    }

    @Test
    public void sourceShapeCanScoreWithoutATextStoreOrRawData() throws Exception {
        Path root = Files.createTempDirectory("region-budget-ss-");
        Path h = original(root), q = query(root, "queries.csv", "0,1,0,1\n");
        Path shape = Paths.get(RegionBudgetQueryExpTest.class.getResource("/source_shape").toURI());
        String[] row = run(root, "ss", "source_shape", "-answerHist", h.toString(), "-queryFile", q.toString(),
                "-shapeDir", shape.toString(), "-shapePacket", shape.resolve("children.bin").toString(),
                "-cutoff", "Infinity", "-scoreOnly", "true");
        assertEquals(2.5, value(row, 2), 0);
        assertEquals(2.5, value(row, 3), 0);
        assertTrue(Double.isFinite(value(row, 6)));
    }

    @Test
    public void featureDumpDoesNotConsultTruthAndIncludesTheRegisteredFullV1Inputs() throws Exception {
        Path root = Files.createTempDirectory("region-budget-features-");
        Path h = original(root);
        Files.write(root.resolve("src_0.txt"), "1\n0,2\n0,2\n10\n".getBytes(StandardCharsets.UTF_8));
        Path first = query(root, "one.csv", ".2,.7,.3,.8,1\n");
        Path second = query(root, "two.csv", ".2,.7,.3,.8,999\n");
        Path fa = root.resolve("first-features.csv"), fb = root.resolve("second-features.csv");
        run(root, "features1", "v1_features", "-answerHist", h.toString(), "-sourceDir", root.toString(),
                "-queryFile", first.toString(), "-featureOut", fa.toString());
        run(root, "features2", "v1_features", "-answerHist", h.toString(), "-sourceDir", root.toString(),
                "-queryFile", second.toString(), "-featureOut", fb.toString());
        assertEquals(Files.readAllLines(fa), Files.readAllLines(fb));
        String header = Files.readAllLines(fa).get(0);
        assertTrue(header.contains(",src_est,")); assertTrue(header.contains(",src_log_est,"));
        assertTrue(header.contains(",skew_pred,")); assertTrue(header.contains(",abs_merge_shift"));
    }

    @Test
    public void savedV1UsesOnlySelectedHAndSourcesWithRealFallback() throws Exception {
        Path root = Files.createTempDirectory("region-budget-v1-");
        Path h = original(root), q = query(root, "queries.csv", "0,1,0,1,999\n");
        Files.write(root.resolve("src_0.txt"), "1\n0,2\n0,2\n10\n".getBytes(StandardCharsets.UTF_8));
        Files.write(root.resolve("arm_99.txt"), "unrelated corrupt ARM file must not be read".getBytes(StandardCharsets.UTF_8));
        String stump = "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=0\n"
                + "objective=regression\nfeature_names=src_log_est\n\nTree=0\nnum_leaves=2\nnum_cat=0\n"
                + "split_feature=0\nthreshold=1\ndecision_type=2\nleft_child=-1\nright_child=-2\n"
                + "leaf_value=0.1 0.25\nis_linear=0\n\nend of trees\n";
        Path model = root.resolve("model.txt"); Files.write(model, stump.getBytes(StandardCharsets.UTF_8));
        Files.write(root.resolve("model.properties"), "label.transform=identity\n".getBytes(StandardCharsets.UTF_8));
        raw(root, ".3\t.3\n1\t1\n1.5\t1.5\n");
        String[] accepted = run(root, "ml_accept", "v1", "-answerHist", h.toString(), "-sourceDir", root.toString(),
                "-modelPath", model.toString(), "-queryFile", q.toString(), "-cutoff", ".3", "-scoreOnly", "true");
        assertEquals(.25, value(accepted, 6), 0); assertEquals(2.5, value(accepted, 3), 0);
        assertEquals(.5, value(accepted, 1), 0); assertEquals("true", accepted[4]);
        String[] rejected = run(root, "ml_reject", "v1", "-answerHist", h.toString(), "-sourceDir", root.toString(),
                "-modelPath", model.toString(), "-queryFile", q.toString(), "-cutoff", ".2",
                "-dataDir", root.toString(), "-numFile", "1");
        assertEquals(2, value(rejected, 3), 0); assertEquals("false", rejected[4]);
        assertTrue(value(rejected, 10) > 0);
    }
}
