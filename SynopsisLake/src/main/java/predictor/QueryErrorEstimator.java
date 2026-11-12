package predictor;

/** Estimated relative error of a range-count answer from one merged synopsis. */
public interface QueryErrorEstimator {

	/** Larger means less trustworthy. Comparable across queries on the same synopsis. */
	double estimate(double minLon, double maxLon, double minLat, double maxLat);
}
