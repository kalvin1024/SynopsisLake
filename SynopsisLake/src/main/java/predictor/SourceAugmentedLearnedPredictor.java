package predictor;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Properties;

/**
 * Existing learned predictor plus the two explicit source-reference inputs.
 * Base inputs use the unchanged FeatureRegistry. This adapter is not a new
 * learner and does not fit, calibrate or alter a model. The caller must pair
 * the model with the same native/child artifacts used to construct its inputs.
 */
public final class SourceAugmentedLearnedPredictor implements ErrorPredictor {
    public static final String LOG_REFERENCE = "critic_log_reference";
    public static final String PROXY_SCORE = "critic_proxy_score";
    private final LightGBMModel model;
    private final SourceShapePredictor source;
    private final String[] names;
    private final FeatureRegistry.Feature[] base;
    private final boolean logLabel;
    private final Double cutoff;

    public SourceAugmentedLearnedPredictor(LightGBMModel model, Properties config, SourceShapePredictor source) {
        if (source == null) throw new IllegalArgumentException("Augmented model requires its source reference");
        this.model = model;
        this.source = source;
        this.names = model.getFeatureNames();
        this.base = new FeatureRegistry.Feature[names.length];
        boolean hasReference = false;
        for (int i = 0; i < names.length; ++i) {
            if (names[i].equals(LOG_REFERENCE) || names[i].equals(PROXY_SCORE)) hasReference = true;
            else base[i] = FeatureRegistry.resolve(new String[] {names[i]})[0];
        }
        if (!hasReference) throw new IllegalArgumentException("Augmented model names no critic reference input");
        String kind = config.getProperty("critic.reference");
        if (kind != null && !kind.trim().equals("source_shape"))
            throw new IllegalArgumentException("Model declares a different critic.reference: " + kind);
        String transform = config.getProperty(LearnedErrorPredictor.LABEL_TRANSFORM, "log").trim();
        if (!transform.equals("log") && !transform.equals("identity"))
            throw new IllegalArgumentException("label.transform must be log or identity");
        this.logLabel = transform.equals("log");
        String value = config.getProperty(LearnedErrorPredictor.GATE_CUTOFF);
        this.cutoff = value == null ? null : Double.valueOf(CutoffTable.parseNumber(value));
    }

    public static SourceAugmentedLearnedPredictor load(String modelPath, String propertiesPath,
                                                       SourceShapePredictor source) throws IOException {
        if (propertiesPath == null) {
            File candidate = new File(new File(modelPath).getAbsoluteFile().getParentFile(), "model.properties");
            if (candidate.isFile()) propertiesPath = candidate.getPath();
        }
        Properties config = new Properties();
        if (propertiesPath != null) {
            try (InputStream stream = new FileInputStream(propertiesPath)) { config.load(stream); }
        }
        return new SourceAugmentedLearnedPredictor(LightGBMModel.load(modelPath), config, source);
    }

    public ErrorGate getGate() { return cutoff == null ? null : new ErrorGate(cutoff); }

    @Override
    public boolean needsBeta() {
        return Arrays.asList(names).contains("skew_pred") || Arrays.asList(names).contains("log_alpha");
    }

    @Override
    public boolean scoreIsRelativeError() { return true; }

    @Override
    public QueryErrorEstimator bind(final SynopsisContext synopsis) {
        return (minLon, maxLon, minLat, maxLat) -> {
            QueryFeatures q = new QueryFeatures(synopsis, minLon, maxLon, minLat, maxLat);
            double reference = source.referenceCount(minLon, maxLon, minLat, maxLat);
            double answer = synopsis.mergedHist.RangeQuery(minLon, maxLon, minLat, maxLat);
            double[] values = new double[names.length];
            for (int i = 0; i < names.length; ++i) {
                if (names[i].equals(LOG_REFERENCE)) values[i] = Math.log1p(reference);
                else if (names[i].equals(PROXY_SCORE)) values[i] = SourceShapePredictor.score(answer, reference);
                else values[i] = base[i].value(q);
            }
            double value = model.predict(values);
            return logLabel ? Math.exp(value) : value;
        };
    }
}
