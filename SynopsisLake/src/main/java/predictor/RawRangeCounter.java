package predictor;

/** Exact range count from the raw data: the fallback when the synopsis is not trusted for a query. */
public interface RawRangeCounter {

	double count(double minLon, double maxLon, double minLat, double maxLat);
}
