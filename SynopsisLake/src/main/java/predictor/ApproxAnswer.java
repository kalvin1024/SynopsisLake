package predictor;

/** One answered range-count query: the answer, how it was produced, and what the gate saw. */
public class ApproxAnswer {

	/** The count returned to the caller: the synopsis estimate, or the exact count when the gate said raw. */
	public final double answer;
	/** The synopsis estimate (always computed, so callers can compare). */
	public final double synopsisEstimate;
	/** true when the answer came from the synopsis; false when it was read from the raw data. */
	public final boolean fromSynopsis;
	/** The predictor's score for this query on this synopsis. */
	public final double errorScore;
	/** The cutoff the score was compared with. */
	public final double cutoff;
	/**
	 * Estimated relative error of the answer: the predictor's estimate when the answer is from the synopsis and
	 * the predictor outputs relative errors (learned models), 0 for a raw answer, NaN otherwise.
	 */
	public final double estimatedRelError;
	public final long gateNanos;
	public final long synopsisNanos;
	/** 0 when the raw data was not read. */
	public final long rawNanos;

	ApproxAnswer(double answer, double synopsisEstimate, boolean fromSynopsis, double errorScore, double cutoff,
			double estimatedRelError, long gateNanos, long synopsisNanos, long rawNanos) {
		this.answer = answer;
		this.synopsisEstimate = synopsisEstimate;
		this.fromSynopsis = fromSynopsis;
		this.errorScore = errorScore;
		this.cutoff = cutoff;
		this.estimatedRelError = estimatedRelError;
		this.gateNanos = gateNanos;
		this.synopsisNanos = synopsisNanos;
		this.rawNanos = rawNanos;
	}
}
