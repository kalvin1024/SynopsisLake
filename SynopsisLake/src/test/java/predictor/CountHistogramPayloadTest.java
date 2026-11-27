package predictor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;

public class CountHistogramPayloadTest {
    private static Path fine(long... counts) throws IOException {
        int n = (int) Math.sqrt(counts.length);
        ByteBuffer b = ByteBuffer.allocate(64 + 4 * counts.length).order(ByteOrder.LITTLE_ENDIAN);
        b.put("SLCNT001".getBytes(StandardCharsets.US_ASCII)).putInt(n).putInt(1);
        b.putDouble(0).putDouble(n).putDouble(0).putDouble(n);
        b.position(64);
        for (long value : counts) b.putInt((int) value);
        Path p = Files.createTempFile("fine-count-", ".bin");
        Files.write(p, b.array());
        return p;
    }

    @Test
    public void exactCountPayloadPreservesUnsignedCountsAndChargesFullFile() throws Exception {
        CountHistogramPayload h = CountHistogramPayload.load(fine(4294967295L).toString());
        assertEquals(4294967295L, h.count(0, 1, 0, 1), 0);
        assertEquals(68, h.serializedBytes());
        assertEquals(0, h.relativeErrorBound(0, 1, 0, 1), 0);
    }

    @Test
    public void conservativeIntervalsCoverInclusiveBoundaryAndPointQueries() throws Exception {
        CountHistogramPayload h = CountHistogramPayload.load(fine(1, 2, 3, 4).toString());
        PointScanCounter raw = new PointScanCounter();
        for (int x = 0; x < 2; x++) for (int y = 0; y < 2; y++)
            for (int k = 0; k < 1 + x * 2 + y; k++) raw.add(x, y);
        double[] b = {-1, 0, .5, 1, 1.5, 2, 3};
        for (int a = 0; a < b.length; a++) for (int c = a; c < b.length; c++)
            for (int d = 0; d < b.length; d++) for (int e = d; e < b.length; e++) {
                double[] interval = h.countInterval(b[a], b[c], b[d], b[e]);
                double truth = raw.count(b[a], b[c], b[d], b[e]);
                assertTrue(interval[0] <= truth);
                assertTrue(truth <= interval[1]);
            }
        double[] quarter = h.countInterval(0, 1, 0, 1);
        assertEquals(1, quarter[0], 0);
        assertEquals(10, quarter[1], 0); // Adjacent bins may contain atoms on the inclusive boundary.
        assertEquals(.9, h.relativeErrorBound(0, 1, 0, 1), 1e-12);
        assertEquals(1, h.relativeErrorBound(1, 1, 0, 2), 0);
    }

    @Test
    public void zeroLowerBoundIsNotSilentlyReplacedWithAPseudocount() throws Exception {
        CountHistogramPayload h = CountHistogramPayload.load(fine(1, 2, 3, 4).toString());
        assertEquals(.25, h.count(.25, .75, .25, .75), 1e-12);
        assertEquals(Double.POSITIVE_INFINITY, h.relativeErrorBound(.25, .75, .25, .75), 0);
        assertEquals(0, h.relativeErrorBound(-3, -2, -3, -2), 0);
        assertEquals(1, CountHistogramPayload.relativeErrorBound(0, 0, 8), 0);
    }

    @Test
    public void malformedAndNonfiniteCountPayloadsAreRejected() throws Exception {
        Path p = fine(1, 2, 3, 4);
        byte[] good = Files.readAllBytes(p), bad = good.clone();
        bad[48] = 1;
        Files.write(p, bad);
        try { CountHistogramPayload.load(p.toString()); fail("reserved bytes accepted"); }
        catch (IOException expected) { assertTrue(expected.getMessage().contains("reserved")); }
        Files.write(p, java.util.Arrays.copyOf(good, good.length + 1));
        try { CountHistogramPayload.load(p.toString()); fail("trailing byte accepted"); }
        catch (IOException expected) { assertTrue(expected.getMessage().contains("invalid")); }
        ByteBuffer.wrap(good).order(ByteOrder.LITTLE_ENDIAN).putDouble(16, Double.NaN);
        Files.write(p, good);
        try { CountHistogramPayload.load(p.toString()); fail("nonfinite root accepted"); }
        catch (IOException expected) { assertTrue(expected.getMessage().contains("invalid")); }
    }

    @Test
    public void existingSlhistCodecLoadsWithoutTextSourceDependencies() throws Exception {
        ByteBuffer b = ByteBuffer.allocate(24 + 8 * 10).order(ByteOrder.LITTLE_ENDIAN);
        b.put("SLHIST01".getBytes(StandardCharsets.US_ASCII)).putInt(2).putInt(2).putDouble(.5);
        for (double v : new double[] {0, 1, 3, 0, 1, 2, 1, 2, 3.5, 4.5}) b.putDouble(v);
        Path p = Files.createTempFile("original-H-", ".hist");
        Files.write(p, b.array());
        CountHistogramPayload h = CountHistogramPayload.load(p.toString());
        assertEquals(.5, h.mergeRate(), 0);
        assertEquals(104, h.serializedBytes());
        assertEquals(11, h.count(0, 3, 0, 2), 0);
        assertEquals(1, h.count(0, 1, 0, 1), 0);
        try { h.countInterval(0, 3, 0, 2); fail("fractional H used as exact fine counts"); }
        catch (IllegalStateException expected) { assertTrue(expected.getMessage().contains("SLCNT001")); }
    }
}
