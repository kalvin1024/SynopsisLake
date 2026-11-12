package predictor;

import org.apache.commons.math3.distribution.BetaDistribution;

import histogram.SimpleSpatialHistogramOpt;
import utils.UtilsFunctionHistogram;

/**
 * The beta distribution Skewness-Align fits to a histogram, with the parameters the experiments use
 * (scale factor 0.5, varepsilon 1e-6). Only needed by Skewness-Align itself and by learned models that take
 * skew_pred or log_alpha as an input.
 */
public final class SkewnessBeta {

	public static final double SCALE_FACTOR = 0.5;
	public static final double VAREPSILON = 1e-6;

	private SkewnessBeta() {
	}

	public static BetaDistribution of(SimpleSpatialHistogramOpt hist) {
		return UtilsFunctionHistogram.ComputeBetaDist(hist.getData(), hist.getLonBoundary(), hist.getLatBoundary(),
				SCALE_FACTOR, VAREPSILON);
	}
}
