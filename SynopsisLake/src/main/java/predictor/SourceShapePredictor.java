package predictor;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.zip.CRC32;

import org.apache.commons.math3.special.Beta;
import org.apache.commons.math3.special.Erf;

/**
 * Training-free R10 assessor of an unchanged merged histogram answer.
 *
 * Reads only the existing SLHIST01 / SLMOM001 native source summaries and an
 * optional SLHIE001 child-moment packet. No raw points, labels, query IDs or
 * saved query scores are accepted by this predictor. Native identity, CRC,
 * dimensions, budgets, masses and moment bytes are checked before use.
 *
 * Statistical policy follows the frozen R08/R10 Python implementation: decoded
 * moments and local rectangles round through float32; Beta marginals use fixed
 * clamps; Gaussian copulas use the fixed sixteen-node integral. This is an
 * approximate reference, not an error certificate or a new distribution model.
 */
public final class SourceShapePredictor implements ErrorPredictor {
    private static final double EPSILON = 1e-6;
    private static final double RHO_LIMIT = .995;
    private static final long UINT32_MAX = 0xffffffffL;
    private static final double[] NODES = {
        -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.755404408355003,
        -0.6178762444026438, -0.45801677765722737, -0.2816035507792589, -0.09501250983763744,
         0.09501250983763744, 0.2816035507792589, 0.45801677765722737, 0.6178762444026438,
         0.755404408355003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499
    };
    private static final double[] WEIGHTS = {
        0.027152459411754176, 0.062253523938647456, 0.0951585116824926, 0.12462897125553407,
        0.1495959888165767, 0.16915651939500265, 0.18260341504492364, 0.18945061045506864,
        0.18945061045506864, 0.18260341504492364, 0.16915651939500265, 0.1495959888165767,
        0.12462897125553407, 0.0951585116824926, 0.062253523938647456, 0.027152459411754176
    };

    private static final class NativeSource {
        int nx, ny;
        double rate;
        double[] bx, by;
        long[] mass;
        byte[] moments;
    }

    private static final class Cell {
        final double[] bounds, moments;
        final long mass;
        long[] children;
        double[][] childMoments;
        Cell(double[] bounds, long mass, byte[] encoded, int offset) throws IOException {
            this.bounds = bounds;
            this.mass = mass;
            this.moments = decodeMoments(encoded, offset);
        }
    }

    private final Cell[] cells;
    private final boolean uniformChildren;
    private final long serializedBytes;
    private final int selectedCells;

    private SourceShapePredictor(Cell[] cells, boolean uniformChildren, long bytes, int selected) {
        this.cells = cells;
        this.uniformChildren = uniformChildren;
        this.serializedBytes = bytes;
        this.selectedCells = selected;
    }

    /** A null packet uses the native R08 reference without child refinements. */
    public static SourceShapePredictor load(String directory, String packetFile) throws IOException {
        File[] files = new File(directory).listFiles((d, name) -> name.matches("src_\\d+\\.hist"));
        require(files != null && files.length > 0, "No native src_<id>.hist files in " + directory);
        Arrays.sort(files, Comparator.comparingInt(SourceShapePredictor::sourceId));
        List<NativeSource> sources = new ArrayList<NativeSource>();
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IOException("SHA-256 is unavailable", e);
        }
        digest.update(le(4).putInt(files.length).array());
        long bytes = 0;
        int totalCells = 0;
        for (File file : files) {
            int sid = sourceId(file);
            byte[] histogram = Files.readAllBytes(file.toPath());
            byte[] moments = Files.readAllBytes(Paths.get(directory, "src_" + sid + "-moments.bin"));
            NativeSource source = readNative(histogram, moments, sid);
            sources.add(source);
            bytes += histogram.length + moments.length;
            try {
                totalCells = Math.addExact(totalCells, source.mass.length);
            } catch (ArithmeticException e) {
                throw new IOException("Too many native cells for this JVM", e);
            }
            digest.update(le(16).putInt(source.nx).putInt(source.ny).putDouble(source.rate).array());
            for (double value : source.bx) digest.update(le(8).putDouble(value).array());
            for (double value : source.by) digest.update(le(8).putDouble(value).array());
            for (long value : source.mass) digest.update(le(8).putLong(value).array());
            digest.update(source.moments);
        }
        byte[] identity = Arrays.copyOf(digest.digest(), 16);
        Cell[] byNativeId = new Cell[totalCells];
        List<Cell> active = new ArrayList<Cell>();
        int global = 0;
        for (NativeSource source : sources) {
            for (int i = 0; i < source.mass.length; ++i, ++global) {
                // Decode every cell, including empty cells, to reject reserved moment bytes.
                decodeMoments(source.moments, 5 * i);
                if (source.mass[i] == 0) continue;
                int ix = i / source.ny, iy = i % source.ny;
                Cell cell = new Cell(new double[] {source.bx[ix], source.bx[ix + 1],
                        source.by[iy], source.by[iy + 1]}, source.mass[i], source.moments, 5 * i);
                byNativeId[global] = cell;
                active.add(cell);
            }
        }
        int selected = 0;
        boolean uniform = false;
        if (packetFile != null) {
            byte[] packet = Files.readAllBytes(Paths.get(packetFile));
            require(packet.length >= 64, "Truncated hierarchical packet");
            ByteBuffer b = wrap(packet);
            require(magic(b, "SLHIE001"), "Unsupported source-shape packet magic");
            int version = Short.toUnsignedInt(b.getShort()), mode = Short.toUnsignedInt(b.getShort());
            int sourceCount = positiveInt(b.getInt(), "packet source count");
            int nativeCount = nonnegativeInt(b.getInt(), "packet native count");
            selected = nonnegativeInt(b.getInt(), "packet selected count");
            int bitmapBytes = nonnegativeInt(b.getInt(), "packet bitmap size");
            long budget = Integer.toUnsignedLong(b.getInt());
            byte[] fingerprint = new byte[16]; b.get(fingerprint);
            byte[] reserved = new byte[12]; b.get(reserved);
            long crc = Integer.toUnsignedLong(b.getInt());
            long expected = 64L + 8L * sourceCount + bitmapBytes + 32L * selected;
            CRC32 checksum = new CRC32(); checksum.update(packet, 0, 60); checksum.update(packet, 64, packet.length - 64);
            require(version == 1 && (mode == 0 || mode == 1) && allZero(reserved)
                    && sourceCount == sources.size() && nativeCount == totalCells
                    && bitmapBytes == (totalCells + 7L) / 8 && expected == packet.length
                    && packet.length <= budget && Arrays.equals(identity, fingerprint)
                    && checksum.getValue() == crc,
                    "Invalid hierarchical version, identity, dimensions, length, budget or CRC");
            int[] expectedPerSource = new int[sourceCount];
            for (int s = 0; s < sourceCount; ++s) {
                int size = nonnegativeInt(b.getInt(), "source table size");
                expectedPerSource[s] = nonnegativeInt(b.getInt(), "selected source count");
                require(size == sources.get(s).mass.length, "Packet source table does not match native grid");
            }
            byte[] bitmap = new byte[bitmapBytes]; b.get(bitmap);
            if (totalCells % 8 != 0) {
                require((Byte.toUnsignedInt(bitmap[bitmap.length - 1]) >>> (totalCells % 8)) == 0,
                        "Nonzero hierarchical bitmap padding");
            }
            int[] actualPerSource = new int[sourceCount];
            int found = 0, source = 0, sourceEnd = sources.get(0).mass.length;
            for (int id = 0; id < totalCells; ++id) {
                while (id >= sourceEnd) sourceEnd += sources.get(++source).mass.length;
                if ((Byte.toUnsignedInt(bitmap[id / 8]) & (1 << (id % 8))) == 0) continue;
                require(found < selected && byNativeId[id] != null, "Invalid selected native cell");
                Cell cell = byNativeId[id];
                cell.children = new long[4];
                long sum = 0;
                for (int k = 0; k < 3; ++k) {
                    cell.children[k] = Integer.toUnsignedLong(b.getInt());
                    sum += cell.children[k];
                }
                require(sum <= cell.mass && cell.mass - sum <= UINT32_MAX, "Child masses exceed native mass");
                cell.children[3] = cell.mass - sum;
                cell.childMoments = new double[4][];
                for (int k = 0; k < 4; ++k) {
                    byte[] encoded = new byte[5]; b.get(encoded);
                    cell.childMoments[k] = decodeMoments(encoded, 0);
                }
                ++found; ++actualPerSource[source];
            }
            require(found == selected && Arrays.equals(actualPerSource, expectedPerSource) && !b.hasRemaining(),
                    "Hierarchical bitmap, records or source counts disagree");
            bytes += packet.length;
            uniform = mode == 1;
        }
        return new SourceShapePredictor(active.toArray(new Cell[0]), uniform, bytes, selected);
    }

    @Override
    public QueryErrorEstimator bind(final SynopsisContext synopsis) {
        return (minLon, maxLon, minLat, maxLat) -> {
            double reference = referenceCount(minLon, maxLon, minLat, maxLat);
            double answer = synopsis.mergedHist.RangeQuery(minLon, maxLon, minLat, maxLat);
            return score(answer, reference);
        };
    }

    @Override
    public boolean scoreIsRelativeError() { return true; }

    public long serializedBytes() { return serializedBytes; }
    public int selectedCells() { return selectedCells; }

    public static double score(double answer, double reference) {
        if (!Double.isFinite(answer) || !Double.isFinite(reference) || answer < 0 || reference < 0)
            throw new IllegalArgumentException("Expected finite nonnegative count estimates");
        return Math.abs(answer - reference) / Math.max(reference, 1.0);
    }

    /** Integrate saved source shapes. This routine never accesses the candidate H. */
    public double referenceCount(double x0, double x1, double y0, double y1) {
        if (!Double.isFinite(x0) || !Double.isFinite(x1) || !Double.isFinite(y0) || !Double.isFinite(y1)
                || x0 > x1 || y0 > y1)
            throw new IllegalArgumentException("Expected finite ordered query bounds");
        long fullMass = 0;
        double boundary = 0;
        for (Cell cell : cells) {
            double[] b = cell.bounds;
            if (x0 > b[1] || x1 < b[0] || y0 > b[3] || y1 < b[2]) continue;
            if (x0 <= b[0] && x1 >= b[1] && y0 <= b[2] && y1 >= b[3]) {
                fullMass = Math.addExact(fullMass, cell.mass);
                continue;
            }
            double[] r = {f32((x0 - b[0]) / (b[1] - b[0])), f32((x1 - b[0]) / (b[1] - b[0])),
                          f32((y0 - b[2]) / (b[3] - b[2])), f32((y1 - b[2]) / (b[3] - b[2]))};
            boundary += cell.children == null ? cell.mass * probability(r, cell.moments) : children(r, cell);
        }
        double answer = fullMass + boundary;
        if (!Double.isFinite(answer)) throw new ArithmeticException("Nonfinite source-shape reference");
        return answer;
    }

    private double children(double[] r, Cell cell) {
        double result = 0;
        for (int k = 0; k < 4; ++k) {
            if (cell.children[k] == 0) continue;
            double x0 = k >= 2 ? .5 : 0, x1 = x0 + .5;
            double y0 = k % 2 == 1 ? .5 : 0, y1 = y0 + .5;
            boolean full = r[0] <= x0 && r[1] >= x1 && r[2] <= y0 && r[3] >= y1;
            double p = full ? 1 : 0;
            double lx = Math.max(r[0], x0), ux = Math.min(r[1], x1);
            double ly = Math.max(r[2], y0), uy = Math.min(r[3], y1);
            if (!full && lx < ux && ly < uy) {
                double[] child = {f32(2 * (lx - x0)), f32(2 * (ux - x0)),
                                  f32(2 * (ly - y0)), f32(2 * (uy - y0))};
                p = uniformChildren ? (child[1] - child[0]) * (child[3] - child[2])
                                    : probability(child, cell.childMoments[k]);
            }
            result += cell.children[k] * p;
        }
        return result;
    }

    private static double[] marginal(double lo, double hi, double mean, double variance) {
        if (variance == 0) return new double[] {lo > mean ? 1 : 0, hi >= mean ? 1 : 0};
        double m = clip(mean, EPSILON, 1 - EPSILON);
        double v = Math.min(variance, m * (1 - m) * (1 - EPSILON));
        double k = m * (1 - m) / v - 1;
        return new double[] {clip(Beta.regularizedBeta(lo, m * k, (1 - m) * k)),
                             clip(Beta.regularizedBeta(hi, m * k, (1 - m) * k))};
    }

    private static double probability(double[] r, double[] m) {
        double[] x = marginal(r[0], r[1], m[0], m[2]), y = marginal(r[2], r[3], m[1], m[3]);
        double denominator = Math.sqrt(m[2] * m[3]);
        double rho = clip(denominator > 0 ? m[4] / denominator : 0, -RHO_LIMIT, RHO_LIMIT);
        double p = copula(x[1], y[1], rho) - copula(x[0], y[1], rho)
                 - copula(x[1], y[0], rho) + copula(x[0], y[0], rho);
        if (!Double.isFinite(p)) throw new ArithmeticException("Nonfinite source-shape probability");
        return clip(p);
    }

    private static double copula(double u, double v, double rho) {
        double result = u * v;
        if (u > 0 && u < 1 && v > 0 && v < 1 && rho != 0) {
            double h = inverseNormal(u), k = inverseNormal(v);
            double angle = Math.asin(rho);
            double[] terms = new double[16];
            for (int i = 0; i < 16; ++i) {
                double t = Math.sin(.5 * angle * (NODES[i] + 1));
                double difference = h - k * t;
                terms[i] = Math.exp(-.5 * (difference * difference / (1 - t * t) + k * k)) * WEIGHTS[i];
            }
            double[] partial = new double[8];
            for (int i = 0; i < 8; ++i) partial[i] = terms[i] + terms[i + 8];
            double integral = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
                            + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
            result += .5 * angle * integral / (2 * Math.PI);
        }
        double upper = Math.min(u, v), lower = Math.min(Math.max(u + v - 1, 0), upper);
        return Math.max(Math.min(result, upper), lower);
    }

    /**
     * Acklam's inverse-normal rational approximation with a stable erfc-based
     * Halley refinement. Avoids erfInv(2*p-1), whose argument rounds to -1 for
     * tiny positive p. No marginal probabilities are clamped or redefined.
     */
    static double inverseNormal(double p) {
        if (Double.isNaN(p) || p < 0 || p > 1) throw new IllegalArgumentException("Invalid normal probability");
        if (p == 0) return Double.NEGATIVE_INFINITY;
        if (p == 1) return Double.POSITIVE_INFINITY;
        if (p > .5) return -inverseNormal(1 - p);
        double x;
        if (p < .02425) {
            double q = Math.sqrt(-2 * Math.log(p));
            x = (((((-.007784894002430293 * q - .3223964580411365) * q - 2.400758277161838)
                    * q - 2.549732539343734) * q + 4.374664141464968) * q + 2.938163982698783)
                  / ((((.007784695709041462 * q + .3224671290700398) * q + 2.445134137142996)
                    * q + 3.754408661907416) * q + 1);
        } else {
            double q = p - .5, r = q * q;
            x = (((((-39.69683028665376 * r + 220.9460984245205) * r - 275.9285104469687)
                    * r + 138.357751867269) * r - 30.66479806614716) * r + 2.506628277459239) * q
                  / (((((-54.47609879822406 * r + 161.5858368580409) * r - 155.6989798598866)
                    * r + 66.80131188771972) * r - 13.28068155288572) * r + 1);
        }
        double density = Math.exp(-.5 * x * x) / Math.sqrt(2 * Math.PI);
        double update = (.5 * Erf.erfc(-x / Math.sqrt(2)) - p) / density;
        if (Double.isFinite(update)) x -= update / (1 + .5 * x * update);
        return x;
    }

    private static NativeSource readNative(byte[] hist, byte[] moments, int sid) throws IOException {
        require(hist.length >= 24, "Truncated native histogram");
        ByteBuffer h = wrap(hist);
        require(magic(h, "SLHIST01"), "Unsupported native histogram magic");
        NativeSource s = new NativeSource();
        s.nx = positiveInt(h.getInt(), "native nx"); s.ny = positiveInt(h.getInt(), "native ny");
        s.rate = h.getDouble();
        long size = (long) s.nx * s.ny;
        require(Double.isFinite(s.rate) && s.rate >= 0 && size <= Integer.MAX_VALUE
                && hist.length == 24L + 8L * (s.nx + (long) s.ny + 2 + size), "Invalid native dimensions or rate");
        s.bx = new double[s.nx + 1]; s.by = new double[s.ny + 1]; s.mass = new long[(int) size];
        for (int i = 0; i < s.bx.length; ++i) s.bx[i] = h.getDouble();
        for (int i = 0; i < s.by.length; ++i) s.by[i] = h.getDouble();
        validateAxis(s.bx); validateAxis(s.by);
        for (int i = 0; i < s.mass.length; ++i) {
            double count = h.getDouble();
            require(Double.isFinite(count) && count >= 0 && count <= 9007199254740992.0 && count == Math.floor(count),
                    "Native masses must be exact nonnegative integers <=2^53");
            s.mass[i] = (long) count;
        }
        require(moments.length == 64L + 5L * size, "Invalid native moment payload length");
        ByteBuffer m = wrap(moments);
        require(magic(m, "SLMOM001"), "Unsupported native moment magic");
        require(Byte.toUnsignedInt(m.get()) == 1 && Byte.toUnsignedInt(m.get()) == 5 && m.getShort() == 0
                && m.getInt() == sid && m.getInt() == s.nx && m.getInt() == s.ny,
                "Native moment version, source ID or dimensions disagree");
        require(m.getDouble() == s.bx[0] && m.getDouble() == s.bx[s.nx]
                && m.getDouble() == s.by[0] && m.getDouble() == s.by[s.ny], "Native moment bounds disagree");
        for (int i = 56; i < 64; ++i) require(moments[i] == 0, "Nonzero native moment padding");
        s.moments = Arrays.copyOfRange(moments, 64, moments.length);
        return s;
    }

    private static void validateAxis(double[] axis) throws IOException {
        for (int i = 0; i < axis.length; ++i)
            require(Double.isFinite(axis[i]) && (i == 0 || (axis[i] > axis[i - 1]
                    && Double.isFinite(axis[i] - axis[i - 1]))), "Invalid native grid boundaries");
    }

    private static double[] decodeMoments(byte[] bytes, int offset) throws IOException {
        require(bytes[offset + 4] != Byte.MIN_VALUE, "Reserved covariance byte -128");
        return new double[] {(double) (float) (Byte.toUnsignedInt(bytes[offset]) / 255.0),
            (double) (float) (Byte.toUnsignedInt(bytes[offset + 1]) / 255.0),
            (double) (float) (Byte.toUnsignedInt(bytes[offset + 2]) / 1020.0),
            (double) (float) (Byte.toUnsignedInt(bytes[offset + 3]) / 1020.0),
            (double) (float) (bytes[offset + 4] / 508.0)};
    }

    private static int sourceId(File file) {
        return Integer.parseInt(file.getName().substring(4, file.getName().length() - 5));
    }
    private static ByteBuffer le(int size) { return ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN); }
    private static ByteBuffer wrap(byte[] bytes) { return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN); }
    private static boolean magic(ByteBuffer b, String expected) {
        for (int i = 0; i < expected.length(); ++i) if (b.get() != expected.charAt(i)) return false;
        return true;
    }
    private static boolean allZero(byte[] bytes) {
        for (byte value : bytes) if (value != 0) return false;
        return true;
    }
    private static int positiveInt(int value, String name) throws IOException {
        require(value > 0, name + " must be a positive int32"); return value;
    }
    private static int nonnegativeInt(int value, String name) throws IOException {
        require(value >= 0, name + " must be a nonnegative int32"); return value;
    }
    private static double clip(double x) { return clip(x, 0, 1); }
    private static double clip(double x, double lo, double hi) { return Math.max(lo, Math.min(x, hi)); }
    private static double f32(double x) { return (double) (float) clip(x); }
    private static void require(boolean condition, String message) throws IOException {
        if (!condition) throw new IOException(message);
    }
}
