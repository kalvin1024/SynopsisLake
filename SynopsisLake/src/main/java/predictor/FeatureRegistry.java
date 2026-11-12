package predictor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Maps the feature names a model was trained on to the Java code that computes them.
 *
 * A learned predictor asks for exactly the names in the model file, in the model's order, so any model built
 * from the registered features works without code changes. A model that names something unknown is rejected
 * when it is loaded. New features are added with {@link #register}; they must give the same value as the
 * training code (ml/features.py, ml/features_v1.py) or the model will mispredict.
 */
public final class FeatureRegistry {

	/** One named model input. */
	public interface Feature {
		double value(QueryFeatures q);
	}

	private static final double E = HistogramView.E;
	private static final Map<String, Feature> FEATURES = new LinkedHashMap<String, Feature>();

	static {
		// query shape (merged histogram)
		register("area_cells", q -> q.merged().areaCells);
		register("n_cells", q -> q.merged().nCells);
		register("n_partial", q -> q.merged().nPartial);
		register("side_frac", q -> q.merged().sideFrac);
		// edge cells
		register("log_est", q -> Math.log1p(q.merged().est));
		register("log_full", q -> Math.log1p(q.merged().fullCount));
		register("bound_rel", q -> q.merged().boundRel());
		register("edge_share", q -> q.merged().edgeShare());
		register("edge_contrast", q -> q.merged().edgeContrast);
		register("in_out", q -> q.merged().inOut);
		register("empty_partial", q -> q.merged().emptyPartial);
		// scale-free
		register("est_frac", q -> q.merged().estFrac);
		register("density_ratio", q -> q.merged().densityRatio);
		// merge
		register("merge_rate", q -> q.ctx.mergeRate);
		register("n_src_overlap", q -> q.sources().nOverlap);
		register("src_res_min", q -> q.sources().resMin);
		register("src_res_max", q -> q.sources().resMax);
		// Skewness-Align's own outputs
		register("skew_pred", q -> q.skewPred());
		register("log_alpha", q -> Math.log10(Math.max(q.ctx.alpha(), 1.0)));
		// source level (v1)
		register("src_est", q -> q.sources().est);
		register("src_log_est", q -> Math.log1p(q.sources().est));
		register("src_bound_rel", q -> q.sources().width / Math.max(q.sources().est, 1.0));
		register("src_edge_share", q -> (q.sources().est - q.sources().fullCount) / Math.max(q.sources().est, E));
		register("src_edge_contrast", q -> q.sources().contrast / Math.max(q.sources().est, E));
		register("src_in_out", q -> q.sources().inOut / Math.max(q.sources().est, E));
		register("src_empty_partial", q -> q.sources().empty / Math.max(q.sources().est, E));
		register("src_n_partial", q -> q.sources().nPartial);
		// merge shift (v1)
		register("merge_shift", q -> q.mergeShift());
		register("abs_merge_shift", q -> Math.abs(q.mergeShift()));
	}

	private FeatureRegistry() {
	}

	public static synchronized void register(String name, Feature feature) {
		FEATURES.put(name, feature);
	}

	public static synchronized boolean has(String name) {
		return FEATURES.containsKey(name);
	}

	public static synchronized List<String> names() {
		return new ArrayList<String>(FEATURES.keySet());
	}

	/** The features for the given names, in that order. Fails listing every name that is not registered. */
	public static synchronized Feature[] resolve(String[] names) {
		Feature[] out = new Feature[names.length];
		List<String> missing = new ArrayList<String>();
		for (int i = 0; i < names.length; i++) {
			out[i] = FEATURES.get(names[i]);
			if (out[i] == null) {
				missing.add(names[i]);
			}
		}
		if (!missing.isEmpty()) {
			throw new IllegalArgumentException("model uses features with no Java implementation: " + missing
					+ " (registered: " + names() + "); add them with FeatureRegistry.register");
		}
		return out;
	}
}
