package predictor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.TreeMap;

import org.apache.commons.math3.distribution.BetaDistribution;

import histogram.SimpleSpatialHistogramOpt;

/**
 * The saved synopses of one run folder: the source histograms (src_&lt;i&gt;.txt) and the merged histogram for each merge
 * rate (arm_&lt;lonRes&gt;.txt). Lets queries be answered from the artifacts alone, without the raw data.
 */
public class SynopsisStore {

	private final SimpleSpatialHistogramOpt[] sources;
	private final TreeMap<Double, SimpleSpatialHistogramOpt> arms; // merge rate -> merged histogram

	private SynopsisStore(SimpleSpatialHistogramOpt[] sources, TreeMap<Double, SimpleSpatialHistogramOpt> arms) {
		this.sources = sources;
		this.arms = arms;
	}

	public static SynopsisStore load(String dir) throws IOException {
		File folder = new File(dir);
		File[] srcFiles = folder.listFiles((d, n) -> n.matches("src_\\d+\\.txt"));
		File[] armFiles = folder.listFiles((d, n) -> n.matches("arm_\\d+\\.txt"));
		if (srcFiles == null || srcFiles.length == 0 || armFiles.length == 0) {
			throw new IOException(dir + ": needs src_<i>.txt and arm_<lonRes>.txt files");
		}
		Arrays.sort(srcFiles, Comparator.comparingInt(f -> index(f.getName())));
		SimpleSpatialHistogramOpt[] sources = new SimpleSpatialHistogramOpt[srcFiles.length];
		for (int i = 0; i < sources.length; i++) {
			sources[i] = HistogramIO.read(srcFiles[i].getPath()).hist;
		}
		TreeMap<Double, SimpleSpatialHistogramOpt> arms = new TreeMap<Double, SimpleSpatialHistogramOpt>();
		for (File f : armFiles) {
			HistogramIO.Loaded l = HistogramIO.read(f.getPath());
			if (arms.put(l.mergeRate, l.hist) != null) {
				throw new IOException(dir + ": two merged histograms for merge rate " + l.mergeRate);
			}
		}
		return new SynopsisStore(sources, arms);
	}

	private static int index(String name) {
		return Integer.parseInt(name.replaceAll("\\D", ""));
	}

	public int numSources() {
		return sources.length;
	}

	/** Merge rates that have a merged histogram, largest (least merged) first. */
	public List<Double> mergeRates() {
		return new ArrayList<Double>(arms.descendingKeySet());
	}

	/** The merge rate closest to the one asked for. */
	public double nearestMergeRate(double mergeRate) {
		double best = Double.NaN;
		for (double r : arms.keySet()) {
			if (Double.isNaN(best) || Math.abs(r - mergeRate) < Math.abs(best - mergeRate)) {
				best = r;
			}
		}
		return best;
	}

	public SimpleSpatialHistogramOpt merged(double mergeRate) {
		SimpleSpatialHistogramOpt h = arms.get(mergeRate);
		if (h == null) {
			throw new IllegalArgumentException("no merged histogram for merge rate " + mergeRate + "; have "
					+ arms.keySet());
		}
		return h;
	}

	/**
	 * The synopsis a predictor binds to. The beta distribution is only computed when withBeta is true (see
	 * {@link ErrorPredictor#needsBeta()}).
	 */
	public SynopsisContext context(double mergeRate, boolean withBeta) {
		SimpleSpatialHistogramOpt arm = merged(mergeRate);
		BetaDistribution beta = withBeta ? SkewnessBeta.of(arm) : null;
		return new SynopsisContext(arm, sources, mergeRate, beta);
	}
}
