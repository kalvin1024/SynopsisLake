package predictor;

import org.apache.commons.math3.distribution.BetaDistribution;

import histogram.SimpleSpatialHistogramOpt;

/**
 * Everything an error predictor may look at for one merged synopsis: the merged histogram, the source
 * histograms it was merged from, the merge rate, and the beta distribution Skewness-Align uses. Nothing here
 * touches raw data.
 *
 * The beta distribution is only needed by Skewness-Align and by learned models that take its output
 * (skew_pred, log_alpha) as an input; it may be null for models that do not.
 */
public class SynopsisContext {

	final SimpleSpatialHistogramOpt mergedHist;
	final BetaDistribution mergedBeta;
	final HistogramView merged;
	final SimpleSpatialHistogramOpt[] sourceHists;
	final HistogramView[] sources;
	final double mergeRate;

	public SynopsisContext(SimpleSpatialHistogramOpt merged, SimpleSpatialHistogramOpt[] sources, double mergeRate,
			BetaDistribution mergedBeta) {
		this.mergedHist = merged;
		this.mergedBeta = mergedBeta;
		this.mergeRate = mergeRate;
		this.sourceHists = sources.clone();
		this.merged = HistogramView.of(merged);
		this.sources = new HistogramView[sources.length];
		for (int i = 0; i < sources.length; i++) {
			this.sources[i] = HistogramView.of(sources[i]);
		}
	}

	/** Skewness-Align error ratio of the merged histogram for one query (answer[2] of RangeQueryWithRatio). */
	double skewnessAlign(double qMinLon, double qMaxLon, double qMinLat, double qMaxLat) {
		if (mergedBeta == null) {
			throw new IllegalStateException("this predictor needs the merged histogram's beta distribution");
		}
		return mergedHist.RangeQueryWithRatio(mergedBeta, qMinLon, qMaxLon, qMinLat, qMaxLat)[2];
	}

	double alpha() {
		if (mergedBeta == null) {
			throw new IllegalStateException("this predictor needs the merged histogram's beta distribution");
		}
		return mergedBeta.getAlpha();
	}
}
