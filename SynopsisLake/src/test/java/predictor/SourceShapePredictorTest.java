package predictor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.junit.Test;

import histogram.SimpleSpatialHistogramOpt;
import sigspatial25.ApproxQueryExp;

/** Golden values are produced by the independent frozen Python implementation. */
public class SourceShapePredictorTest {
    private static Path fixture() throws Exception {
        return Paths.get(SourceShapePredictorTest.class.getResource("/source_shape").toURI());
    }

    private static Path copyFixture() throws Exception {
        Path out = Files.createTempDirectory("source-shape-test-");
        for (String name : new String[] {"src_0.hist", "src_0-moments.bin", "children.bin"})
            Files.copy(fixture().resolve(name), out.resolve(name));
        return out;
    }

    @Test
    public void nativeAndChildrenMatchIndependentGoldenRectangles() throws Exception {
        Path root = fixture();
        SourceShapePredictor nativeOnly = SourceShapePredictor.load(root.toString(), null);
        SourceShapePredictor child = SourceShapePredictor.load(root.toString(), root.resolve("children.bin").toString());
        List<String> rows = Files.readAllLines(root.resolve("golden.csv"), StandardCharsets.UTF_8);
        for (String line : rows.subList(1, rows.size())) {
            String[] parts = line.split(",");
            double[] q = new double[6];
            for (int i = 0; i < q.length; ++i) q[i] = Double.parseDouble(parts[i]);
            assertEquals(q[4], nativeOnly.referenceCount(q[0], q[1], q[2], q[3]), 1e-9);
            assertEquals(q[5], child.referenceCount(q[0], q[1], q[2], q[3]), 1e-9);
        }
        assertEquals(0, nativeOnly.selectedCells());
        assertEquals(2, child.selectedCells());
        assertEquals(Files.size(root.resolve("src_0.hist")) + Files.size(root.resolve("src_0-moments.bin"))
                + Files.size(root.resolve("children.bin")), child.serializedBytes());
        assertEquals(17, child.referenceCount(-1, 3, -1, 3), 0);
    }

    @Test
    public void inverseNormalKeepsTinyInteriorProbabilitiesFinite() {
        double[][] gold = {{1e-20, -9.2623400897984087}, {1e-30, -11.464024688443613},
                           {1e-300, -37.047096299361201}, {.5, 0}, {.25, -.6744897501960817}};
        for (double[] value : gold)
            assertEquals(value[1], SourceShapePredictor.inverseNormal(value[0]), 2e-12);
        assertTrue(Double.isFinite(SourceShapePredictor.inverseNormal(Double.MIN_VALUE)));
        assertEquals(-SourceShapePredictor.inverseNormal(.25), SourceShapePredictor.inverseNormal(.75), 0);
    }

    @Test
    public void corruptionAndNativeIdentityMismatchAreRejected() throws Exception {
        Path root = copyFixture();
        Path packet = root.resolve("children.bin");
        byte[] original = Files.readAllBytes(packet), corrupt = original.clone();
        corrupt[corrupt.length - 1] ^= 1;
        Files.write(packet, corrupt);
        expectBadPacket(root, "corrupted CRC");
        Files.write(packet, original);
        byte[] moments = Files.readAllBytes(root.resolve("src_0-moments.bin"));
        moments[64] ^= 1; // Valid byte, but no longer matches the packet's native identity.
        Files.write(root.resolve("src_0-moments.bin"), moments);
        expectBadPacket(root, "changed native identity");
    }

    @Test
    public void reservedMomentCovarianceAndTrailingPacketBytesAreRejected() throws Exception {
        Path root = copyFixture();
        byte[] moments = Files.readAllBytes(root.resolve("src_0-moments.bin"));
        moments[68] = Byte.MIN_VALUE;
        Files.write(root.resolve("src_0-moments.bin"), moments);
        try {
            SourceShapePredictor.load(root.toString(), null);
            fail("reserved covariance was accepted");
        } catch (IOException expected) {
            assertTrue(expected.getMessage().contains("Reserved covariance"));
        }
        root = copyFixture();
        byte[] old = Files.readAllBytes(root.resolve("children.bin"));
        Files.write(root.resolve("children.bin"), java.util.Arrays.copyOf(old, old.length + 1));
        expectBadPacket(root, "trailing packet byte");
    }

    private static void expectBadPacket(Path root, String context) throws Exception {
        try {
            SourceShapePredictor.load(root.toString(), root.resolve("children.bin").toString());
            fail("accepted " + context);
        } catch (IOException expected) {
            assertTrue(expected.getMessage().contains("Invalid hierarchical"));
        }
    }

    @Test
    public void sharedGateUsesTheUnchangedSynopsisOrRealMutableRawCounter() throws Exception {
        Path root = fixture();
        SourceShapePredictor predictor = SourceShapePredictor.load(root.toString(), root.resolve("children.bin").toString());
        SimpleSpatialHistogramOpt h = new SimpleSpatialHistogramOpt(0, 0, 2, 2, 1, 1);
        h.getData()[0][0] = 17;
        SynopsisContext context = new SynopsisContext(h, new SimpleSpatialHistogramOpt[] {h}, 1, null);
        PointScanCounter raw = new PointScanCounter();
        raw.add(.1, .1); raw.add(.2, .2); raw.add(1.5, 1.5);
        GatedRangeQuery reject = new GatedRangeQuery(context, predictor, Double.NEGATIVE_INFINITY, raw);
        ApproxAnswer before = reject.answer(0, .5, 0, .5);
        assertFalse(before.fromSynopsis);
        assertEquals(2, before.answer, 0);
        raw.add(.3, .3);
        ApproxAnswer after = reject.answer(0, .5, 0, .5);
        assertEquals(3, after.answer, 0); // Changed raw data, unchanged synopsis/reference.
        assertTrue(after.rawNanos > 0);
        GatedRangeQuery accept = new GatedRangeQuery(context, predictor, Double.POSITIVE_INFINITY, raw);
        ApproxAnswer a = accept.answer(0, .5, 0, .5);
        assertTrue(a.fromSynopsis);
        assertEquals(h.RangeQuery(0, .5, 0, .5), a.answer, 0);
        assertEquals(0, a.rawNanos);
    }

    @Test
    public void incompatibleCliPredictorsAreRejectedBeforeReadingArtifacts() throws Exception {
        try {
            ApproxQueryExp.main(new String[] {"-runDir", "missing", "-queryFile", "missing", "-mergeRate", ".5",
                    "-modelPath", "model.txt", "-shapeDir", "native", "-cutoff", ".1"});
            fail("multiple predictor types were accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("mutually exclusive"));
        }
        try {
            ApproxQueryExp.main(new String[] {"-runDir", "missing", "-queryFile", "missing", "-mergeRate", ".5",
                    "-shapePacket", "children.bin", "-cutoff", ".1"});
            fail("packet without native sources was accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("requires -shapeDir"));
        }
    }
}
