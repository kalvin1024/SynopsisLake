package predictor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import histogram.SimpleSpatialHistogramOpt;

/**
 * The existing Python SLHIST01 and SLCNT001 count payloads, read without a
 * directory-wide synopsis lookup. The latter stores exact uint32 bin counts.
 */
public final class CountHistogramPayload {
    private final SimpleSpatialHistogramOpt histogram;
    private final double[] bx, by;
    private final long[][] prefix;
    private final long serializedBytes;
    private final double mergeRate;

    private CountHistogramPayload(double[] bx, double[] by, double[][] counts,
            long[][] prefix, long serializedBytes, double mergeRate) {
        this.bx = bx;
        this.by = by;
        this.prefix = prefix;
        this.serializedBytes = serializedBytes;
        this.mergeRate = mergeRate;
        this.histogram = new SimpleSpatialHistogramOpt(bx[0], by[0], bx[bx.length - 1],
                by[by.length - 1], bx.length - 1, by.length - 1, counts, bx, by);
    }

    public static CountHistogramPayload load(String path) throws IOException {
        byte[] payload = Files.readAllBytes(Paths.get(path));
        if (payload.length < 8) throw new IOException(path + ": truncated count histogram");
        String magic = new String(payload, 0, 8, StandardCharsets.US_ASCII);
        ByteBuffer in = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        in.position(8);
        if (magic.equals("SLHIST01")) return readNative(path, payload.length, in);
        if (magic.equals("SLCNT001")) return readFine(path, payload.length, in);
        throw new IOException(path + ": expected SLHIST01 or SLCNT001 count payload");
    }

    private static CountHistogramPayload readNative(String path, int length, ByteBuffer in) throws IOException {
        if (length < 24) throw new IOException(path + ": truncated SLHIST01 header");
        int nx = in.getInt(), ny = in.getInt();
        double rate = in.getDouble();
        long cells = (long) nx * ny;
        if (nx <= 0 || ny <= 0 || cells > Integer.MAX_VALUE
                || cells + nx + ny + 2 != (length - 24) / 8L || (length - 24) % 8 != 0
                || !Double.isFinite(rate) || rate < 0)
            throw new IOException(path + ": invalid SLHIST01 dimensions/rate");
        double[] bx = new double[nx + 1], by = new double[ny + 1];
        for (int i = 0; i < bx.length; i++) bx[i] = in.getDouble();
        for (int i = 0; i < by.length; i++) by[i] = in.getDouble();
        checkBoundaries(path, bx); checkBoundaries(path, by);
        double[][] counts = new double[nx][ny];
        for (int i = 0; i < nx; i++) for (int j = 0; j < ny; j++) {
            double value = in.getDouble();
            if (!Double.isFinite(value) || value < 0) throw new IOException(path + ": invalid count");
            counts[i][j] = value;
        }
        return new CountHistogramPayload(bx, by, counts, null, length, rate);
    }

    private static CountHistogramPayload readFine(String path, int length, ByteBuffer in) throws IOException {
        if (length < 64) throw new IOException(path + ": truncated SLCNT001 header");
        int n = in.getInt(), version = in.getInt();
        long cells = (long) n * n;
        double x0 = in.getDouble(), x1 = in.getDouble(), y0 = in.getDouble(), y1 = in.getDouble();
        if (n <= 0 || cells > Integer.MAX_VALUE || cells != (length - 64) / 4L
                || (length - 64) % 4 != 0 || version != 1 || !Double.isFinite(x0)
                || !Double.isFinite(x1) || !Double.isFinite(y0) || !Double.isFinite(y1)
                || !(x0 < x1) || !(y0 < y1))
            throw new IOException(path + ": invalid SLCNT001 dimensions/root/version");
        for (int i = 0; i < 16; i++) if (in.get() != 0)
            throw new IOException(path + ": nonzero reserved SLCNT001 header bytes");
        double[] bx = linspace(x0, x1, n), by = linspace(y0, y1, n);
        checkBoundaries(path, bx); checkBoundaries(path, by);
        double[][] counts = new double[n][n];
        long[][] prefix = new long[n + 1][n + 1];
        try {
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
                long value = Integer.toUnsignedLong(in.getInt());
                counts[i][j] = value;
                prefix[i + 1][j + 1] = Math.addExact(value,
                        Math.subtractExact(Math.addExact(prefix[i][j + 1], prefix[i + 1][j]), prefix[i][j]));
            }
        } catch (ArithmeticException e) {
            throw new IOException(path + ": count prefix overflows signed long", e);
        }
        return new CountHistogramPayload(bx, by, counts, prefix, length, 0);
    }

    private static double[] linspace(double lo, double hi, int n) {
        double[] b = new double[n + 1];
        double step = (hi - lo) / n;
        for (int i = 0; i < n; i++) b[i] = lo + i * step;
        b[n] = hi;
        return b;
    }

    private static void checkBoundaries(String path, double[] values) throws IOException {
        for (int i = 0; i < values.length; i++)
            if (!Double.isFinite(values[i]) || (i > 0 && !(values[i - 1] < values[i])))
                throw new IOException(path + ": invalid histogram boundaries");
    }

    public SimpleSpatialHistogramOpt histogram() { return histogram; }
    public long serializedBytes() { return serializedBytes; }
    public double mergeRate() { return mergeRate; }

    public double count(double x0, double x1, double y0, double y1) {
        validateQuery(x0, x1, y0, y1);
        if (x0 == x1 || y0 == y1) return 0;
        if (prefix == null) return histogram.RangeQuery(x0, x1, y0, y1);
        // The fine baseline is a uniform-within-bin count estimate. Sum at most
        // nine weighted count rectangles using its derived integer prefix;
        // original H deliberately keeps the unchanged team RangeQuery path.
        double[][] x = overlapSegments(bx, x0, x1), y = overlapSegments(by, y0, y1);
        double value = 0;
        for (double[] a : x) for (double[] b : y)
            value += a[2] * b[2] * rectangleSum((int) a[0], (int) a[1], (int) b[0], (int) b[1]);
        double[] interval = countInterval(x0, x1, y0, y1);
        return Math.max(interval[0], Math.min(interval[1], value));
    }

    private static double[][] overlapSegments(double[] bounds, double lo, double hi) {
        lo = Math.max(lo, bounds[0]); hi = Math.min(hi, bounds[bounds.length - 1]);
        if (!(lo < hi)) return new double[0][0];
        int first = upperBound(bounds, lo) - 1, end = lowerBound(bounds, hi);
        if (first + 1 == end)
            return new double[][] {{first, end, (hi - lo) / (bounds[end] - bounds[first])}};
        double left = (bounds[first + 1] - lo) / (bounds[first + 1] - bounds[first]);
        double right = (hi - bounds[end - 1]) / (bounds[end] - bounds[end - 1]);
        if (first + 2 == end)
            return new double[][] {{first, first + 1, left}, {end - 1, end, right}};
        return new double[][] {{first, first + 1, left}, {first + 1, end - 1, 1}, {end - 1, end, right}};
    }

    /** [L,U] from fully contained versus possibly intersecting closed bin supports.
     * Touching bins are intentionally included in U for the inclusive raw-query
     * convention. Derived integer prefix sums cost initialization/RAM, no extra
     * serialized artifact, and never consult raw points or labels.
     */
    public double[] countInterval(double x0, double x1, double y0, double y1) {
        validateQuery(x0, x1, y0, y1);
        if (prefix == null) throw new IllegalStateException("Count bounds require an exact SLCNT001 payload");
        int nx = bx.length - 1, ny = by.length - 1;
        int lx0 = clamp(lowerBound(bx, x0), nx), lx1 = clamp(upperBound(bx, x1) - 1, nx);
        int ly0 = clamp(lowerBound(by, y0), ny), ly1 = clamp(upperBound(by, y1) - 1, ny);
        int ux0 = clamp(lowerBound(bx, x0) - 1, nx), ux1 = clamp(upperBound(bx, x1), nx);
        int uy0 = clamp(lowerBound(by, y0) - 1, ny), uy1 = clamp(upperBound(by, y1), ny);
        long lower = rectangleSum(lx0, lx1, ly0, ly1), upper = rectangleSum(ux0, ux1, uy0, uy1);
        return new double[] {lower, upper};
    }

    public double relativeErrorBound(double x0, double x1, double y0, double y1) {
        double answer = count(x0, x1, y0, y1);
        double[] interval = countInterval(x0, x1, y0, y1);
        return relativeErrorBound(answer, interval[0], interval[1]);
    }

    /** Worst relative discrepancy over a nonnegative count interval; infinities
     * are not replaced with a finite constant or accepted by a +Infinity gate.
     */
    public static double relativeErrorBound(double answer, double lower, double upper) {
        if (!Double.isFinite(answer) || !Double.isFinite(lower) || !Double.isFinite(upper)
                || answer < 0 || lower < 0 || upper < lower)
            throw new IllegalArgumentException("Expected finite nonnegative answer and ordered count interval");
        if (lower == 0) return answer > 0 ? Double.POSITIVE_INFINITY : upper > 0 ? 1 : 0;
        return Math.max(Math.abs(answer - lower) / lower, Math.abs(upper - answer) / upper);
    }

    private long rectangleSum(int x0, int x1, int y0, int y1) {
        if (x0 >= x1 || y0 >= y1) return 0;
        return prefix[x1][y1] - prefix[x0][y1] - prefix[x1][y0] + prefix[x0][y0];
    }

    private static int clamp(int x, int n) { return Math.min(Math.max(x, 0), n); }
    private static int lowerBound(double[] a, double x) {
        int lo = 0, hi = a.length;
        while (lo < hi) { int m = (lo + hi) >>> 1; if (a[m] < x) lo = m + 1; else hi = m; }
        return lo;
    }
    private static int upperBound(double[] a, double x) {
        int lo = 0, hi = a.length;
        while (lo < hi) { int m = (lo + hi) >>> 1; if (a[m] <= x) lo = m + 1; else hi = m; }
        return lo;
    }
    private static void validateQuery(double x0, double x1, double y0, double y1) {
        if (!Double.isFinite(x0) || !Double.isFinite(x1) || !Double.isFinite(y0)
                || !Double.isFinite(y1) || x0 > x1 || y0 > y1)
            throw new IllegalArgumentException("Expected finite ordered query bounds");
    }
}
