package predictor;

/**
 * Answer a query from the merged synopsis when its estimated error is at or below the cutoff; otherwise the
 * caller reads the raw data. The cutoff is chosen offline on calibration queries (ml/gate_eval.cutoff) for a
 * target error limit and violation rate.
 */
public class ErrorGate {

	private final double cutoff;

	public ErrorGate(double cutoff) {
		this.cutoff = cutoff;
	}

	public double getCutoff() {
		return cutoff;
	}

	public boolean useSynopsis(double estimatedError) {
		return estimatedError <= cutoff;
	}
}
