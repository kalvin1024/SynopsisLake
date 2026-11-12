package predictor;

/**
 * Decides how far a merged synopsis can be trusted for a query. SkewnessAlignPredictor is the original rule;
 * LearnedErrorPredictor replaces it with a trained model. Callers depend only on this interface.
 */
public interface ErrorPredictor {

	/** Prepares the predictor for one merged synopsis; per-histogram work happens here, once. */
	QueryErrorEstimator bind(SynopsisContext synopsis);

	/** true when bind needs SynopsisContext's beta distribution (Skewness-Align, or a model using skew_pred/log_alpha). */
	default boolean needsBeta() {
		return false;
	}

	/** true when the estimate is a relative error (learned models), false when it is only a score (Skewness-Align). */
	default boolean scoreIsRelativeError() {
		return false;
	}
}
