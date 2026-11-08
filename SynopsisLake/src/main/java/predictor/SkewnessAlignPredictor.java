package predictor;

/** The original Skewness-Align error estimate (answer[2] of SimpleSpatialHistogramOpt.RangeQueryWithRatio). */
public class SkewnessAlignPredictor implements ErrorPredictor {

	@Override
	public boolean needsBeta() {
		return true;
	}

	@Override
	public QueryErrorEstimator bind(final SynopsisContext synopsis) {
		return (minLon, maxLon, minLat, maxLat) -> synopsis.skewnessAlign(minLon, maxLon, minLat, maxLat);
	}
}
