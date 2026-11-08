package predictor;

/**
 * The query path: answers a range count from the merged synopsis when the error predictor trusts it, and from
 * the raw data otherwise. Every answer carries its estimated relative error and the decision.
 *
 * Which predictor and which cutoff are inputs: any {@link ErrorPredictor} (Skewness-Align or a learned model)
 * bound to the synopsis, and a cutoff from {@link CutoffTable} for that synopsis's merge rate.
 *
 * Without a {@link RawRangeCounter} a "raw" decision still returns the synopsis estimate, marked
 * fromSynopsis=false, so callers can see what the gate wanted and do the read themselves.
 */
public class GatedRangeQuery {

	private final SynopsisContext synopsis;
	private final QueryErrorEstimator estimator;
	private final double cutoff;
	private final boolean scoreIsRelativeError;
	private final RawRangeCounter raw;

	public GatedRangeQuery(SynopsisContext synopsis, ErrorPredictor predictor, double cutoff, RawRangeCounter raw) {
		this.synopsis = synopsis;
		this.estimator = predictor.bind(synopsis);
		this.cutoff = cutoff;
		this.scoreIsRelativeError = predictor.scoreIsRelativeError();
		this.raw = raw;
	}

	public double getCutoff() {
		return cutoff;
	}

	public ApproxAnswer answer(double minLon, double maxLon, double minLat, double maxLat) {
		long t0 = System.nanoTime();
		double score = estimator.estimate(minLon, maxLon, minLat, maxLat);
		long t1 = System.nanoTime();
		double estimate = synopsis.mergedHist.RangeQuery(minLon, maxLon, minLat, maxLat);
		long t2 = System.nanoTime();
		if (score <= cutoff) {
			return new ApproxAnswer(estimate, estimate, true, score, cutoff,
					scoreIsRelativeError ? score : Double.NaN, t1 - t0, t2 - t1, 0);
		}
		if (raw == null) {
			return new ApproxAnswer(estimate, estimate, false, score, cutoff, Double.NaN, t1 - t0, t2 - t1, 0);
		}
		long t3 = System.nanoTime();
		double exact = raw.count(minLon, maxLon, minLat, maxLat);
		long t4 = System.nanoTime();
		return new ApproxAnswer(exact, estimate, false, score, cutoff, 0.0, t1 - t0, t2 - t1, t4 - t3);
	}
}
