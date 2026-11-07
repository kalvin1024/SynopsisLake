package predictor;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Error predictor backed by a trained LightGBM regression model (model.txt).
 *
 * The model file decides which features are used and in what order; nothing about them is fixed here. An
 * optional properties file next to the model (model.properties, or the path given) describes what the model
 * predicts:
 *
 *   label.transform   log (default): the model predicts log(relative error), so the estimate is exp(output)
 *                     identity: the model predicts the relative error itself
 *   gate.cutoff       calibrated cutoff for {@link ErrorGate}; optional
 */
public class LearnedErrorPredictor implements ErrorPredictor {

	public static final String LABEL_TRANSFORM = "label.transform";
	public static final String GATE_CUTOFF = "gate.cutoff";

	private final LightGBMModel model;
	private final FeatureRegistry.Feature[] features;
	private final boolean logLabel;
	private final Double cutoff;

	public LearnedErrorPredictor(LightGBMModel model, Properties config) {
		this.model = model;
		this.features = FeatureRegistry.resolve(model.getFeatureNames());
		String transform = config.getProperty(LABEL_TRANSFORM, "log").trim();
		if (!transform.equals("log") && !transform.equals("identity")) {
			throw new IllegalArgumentException(LABEL_TRANSFORM + " must be log or identity, got " + transform);
		}
		this.logLabel = transform.equals("log");
		String c = config.getProperty(GATE_CUTOFF);
		this.cutoff = c == null ? null : Double.valueOf(c.trim());
	}

	/** Loads model.txt and, if present, model.properties from the same folder. */
	public static LearnedErrorPredictor load(String modelPath) throws IOException {
		File sidecar = new File(new File(modelPath).getAbsoluteFile().getParentFile(), "model.properties");
		return load(modelPath, sidecar.exists() ? sidecar.getPath() : null);
	}

	public static LearnedErrorPredictor load(String modelPath, String propertiesPath) throws IOException {
		Properties config = new Properties();
		if (propertiesPath != null) {
			try (InputStream in = new FileInputStream(propertiesPath)) {
				config.load(in);
			}
		}
		return new LearnedErrorPredictor(LightGBMModel.load(modelPath), config);
	}

	public LightGBMModel getModel() {
		return model;
	}

	/** Gate built from gate.cutoff, or null when the properties give none. */
	public ErrorGate getGate() {
		return cutoff == null ? null : new ErrorGate(cutoff);
	}

	@Override
	public QueryErrorEstimator bind(final SynopsisContext synopsis) {
		return (minLon, maxLon, minLat, maxLat) -> {
			QueryFeatures q = new QueryFeatures(synopsis, minLon, maxLon, minLat, maxLat);
			double[] x = new double[features.length];
			for (int i = 0; i < x.length; i++) {
				x[i] = features[i].value(q);
			}
			double y = model.predict(x);
			return logLabel ? Math.exp(y) : y;
		};
	}
}
