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

	private GatedRangeQuery(RawRangeCounter raw) {
		this.synopsis = null;
		this.estimator = null;
		this.cutoff = Double.NaN;
		this.scoreIsRelativeError = false;
		this.raw = raw;
	}

	/**
	 * Exact mode: every query is answered from the raw data. No synopsis, predictor or cutoff is involved; answers
	 * have fromSynopsis=false, estimatedRelError=0, and NaN synopsisEstimate / errorScore / cutoff.
	 */
	public static GatedRangeQuery exact(RawRangeCounter raw) {
		if (raw == null) {
			throw new IllegalArgumentException("exact mode needs a raw data counter");
		}
		return new GatedRangeQuery(raw);
	}

	public double getCutoff() {
		return cutoff;
	}

	public ApproxAnswer answer(double minLon, double maxLon, double minLat, double maxLat) {
		if (synopsis == null) {
			long r0 = System.nanoTime();
			double exact = raw.count(minLon, maxLon, minLat, maxLat);
			long r1 = System.nanoTime();
			return new ApproxAnswer(exact, Double.NaN, false, Double.NaN, Double.NaN, 0.0, 0, 0, r1 - r0);
		}
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
			// the answer is still the synopsis estimate, so its estimated error is the predictor's
			return new ApproxAnswer(estimate, estimate, false, score, cutoff,
					scoreIsRelativeError ? score : Double.NaN, t1 - t0, t2 - t1, 0);
		}
		long t3 = System.nanoTime();
		double exact = raw.count(minLon, maxLon, minLat, maxLat);
		long t4 = System.nanoTime();
		return new ApproxAnswer(exact, estimate, false, score, cutoff, 0.0, t1 - t0, t2 - t1, t4 - t3);
	}
}
