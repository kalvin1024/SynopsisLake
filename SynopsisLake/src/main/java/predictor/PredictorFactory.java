package predictor;

import java.io.IOException;

/** Builds the predictor and cutoffs from input values (a model path, or none for Skewness-Align). */
public final class PredictorFactory {

	private PredictorFactory() {
	}

	/**
	 * @param modelPath      a LightGBM model.txt (model.properties next to it is read if present), or null for the
	 *                       original Skewness-Align rule
	 * @param propertiesPath explicit properties file, or null to use the one next to the model
	 */
	public static ErrorPredictor predictor(String modelPath, String propertiesPath) throws IOException {
		if (modelPath == null) {
			return new SkewnessAlignPredictor();
		}
		return propertiesPath == null ? LearnedErrorPredictor.load(modelPath)
				: LearnedErrorPredictor.load(modelPath, propertiesPath);
	}

	/**
	 * Cutoffs by priority: a cutoffs file (per merge rate), else a single cutoff value, else gate.cutoff from the
	 * model's properties. The single value or gate.cutoff also fills in merge rates a cutoffs file lacks.
	 *
	 * @return null when none of the three is given
	 */
	public static CutoffTable cutoffs(String cutoffsPath, Double cutoff, Double eps, Double delta,
			ErrorPredictor predictor) throws IOException {
		Double fallback = cutoff;
		if (fallback == null && predictor instanceof LearnedErrorPredictor) {
			ErrorGate g = ((LearnedErrorPredictor) predictor).getGate();
			fallback = g == null ? null : g.getCutoff();
		}
		if (cutoffsPath != null) {
			return CutoffTable.load(cutoffsPath, eps, delta, fallback);
		}
		return fallback == null ? null : CutoffTable.constant(fallback);
	}
}
