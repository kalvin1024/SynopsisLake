package predictor;

/**
 * Decides how far a merged synopsis can be trusted for a query. SkewnessAlignPredictor is the original rule;
 * LearnedErrorPredictor replaces it with a trained model. Callers depend only on this interface.
 */
public interface ErrorPredictor {

	/** Prepares the predictor for one merged synopsis; per-histogram work happens here, once. */
	QueryErrorEstimator bind(SynopsisContext synopsis);
}
