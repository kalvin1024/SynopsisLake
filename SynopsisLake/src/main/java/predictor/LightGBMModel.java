package predictor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A LightGBM regression model read from its native text format (Booster.save_model, "model.txt").
 *
 * Scoring follows LightGBM's own tree walk: numerical splits with the default-direction and missing-value
 * rules, categorical bitset splits, and the objective's output transform. Models that are not single-output
 * regression, or that use linear trees, are rejected when loaded instead of scoring wrongly.
 *
 * The feature order is taken from the file (feature_names), so the caller never hard-codes it.
 */
public class LightGBMModel {

	private static final double K_ZERO_THRESHOLD = 1e-35;
	private static final int MISSING_NONE = 0;
	private static final int MISSING_ZERO = 1;
	private static final int MISSING_NAN = 2;

	private static final class Tree {
		int numLeaves;
		int[] splitFeature;
		double[] threshold;
		int[] decisionType;
		int[] leftChild;
		int[] rightChild;
		double[] leafValue;
		int[] catBoundaries;
		int[] catThreshold;
	}

	private enum Transform {
		IDENTITY, EXP, SQUARE
	}

	private final String[] featureNames;
	private final Tree[] trees;
	private final boolean averageOutput;
	private final Transform transform;
	private final String objective;

	private LightGBMModel(String[] featureNames, Tree[] trees, boolean averageOutput, Transform transform,
			String objective) {
		this.featureNames = featureNames;
		this.trees = trees;
		this.averageOutput = averageOutput;
		this.transform = transform;
		this.objective = objective;
	}

	public static LightGBMModel load(String path) throws IOException {
		try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
			return parse(reader);
		}
	}

	public static LightGBMModel fromString(String text) throws IOException {
		return parse(new BufferedReader(new StringReader(text)));
	}

	/** Feature names in the order {@link #predict(double[])} expects its values. */
	public String[] getFeatureNames() {
		return featureNames.clone();
	}

	public int getNumTrees() {
		return trees.length;
	}

	public String getObjective() {
		return objective;
	}

	/** Model output (after the objective's transform) for one row; features in {@link #getFeatureNames()} order. */
	public double predict(double[] features) {
		if (features.length < featureNames.length) {
			throw new IllegalArgumentException(
					"model needs " + featureNames.length + " features, got " + features.length);
		}
		double sum = 0;
		for (Tree t : trees) {
			sum += t.leafValue[leaf(t, features)];
		}
		if (averageOutput && trees.length > 0) {
			sum /= trees.length;
		}
		switch (transform) {
		case EXP:
			return Math.exp(sum);
		case SQUARE:
			return Math.signum(sum) * sum * sum;
		default:
			return sum;
		}
	}

	private static int leaf(Tree t, double[] x) {
		if (t.numLeaves <= 1) {
			return 0;
		}
		int node = 0;
		while (node >= 0) {
			double v = x[t.splitFeature[node]];
			int type = t.decisionType[node];
			boolean left = (type & 1) != 0 ? categoricalLeft(t, node, v, type) : numericalLeft(t, node, v, type);
			node = left ? t.leftChild[node] : t.rightChild[node];
		}
		return ~node;
	}

	private static boolean numericalLeft(Tree t, int node, double v, int type) {
		int missing = (type >> 2) & 3;
		if (Double.isNaN(v) && missing != MISSING_NAN) {
			v = 0.0;
		}
		if ((missing == MISSING_ZERO && v >= -K_ZERO_THRESHOLD && v <= K_ZERO_THRESHOLD)
				|| (missing == MISSING_NAN && Double.isNaN(v))) {
			return (type & 2) != 0;
		}
		return v <= t.threshold[node];
	}

	private static boolean categoricalLeft(Tree t, int node, double v, int type) {
		int missing = (type >> 2) & 3;
		int cat;
		if (Double.isNaN(v)) {
			if (missing == MISSING_NAN) {
				return false;
			}
			cat = 0;
		} else {
			cat = (int) v;
			if (cat < 0) {
				return false;
			}
		}
		int idx = (int) t.threshold[node];
		int begin = t.catBoundaries[idx];
		int end = t.catBoundaries[idx + 1];
		int word = cat >> 5;
		if (word >= end - begin) {
			return false;
		}
		return ((t.catThreshold[begin + word] >>> (cat & 31)) & 1) != 0;
	}

	// parsing

	private static LightGBMModel parse(BufferedReader reader) throws IOException {
		Map<String, String> header = new HashMap<String, String>();
		boolean averageOutput = false;
		List<Tree> trees = new ArrayList<Tree>();
		Map<String, String> block = null;
		String line;
		while ((line = reader.readLine()) != null) {
			line = line.trim();
			if (line.isEmpty()) {
				continue;
			}
			if (line.startsWith("Tree=")) {
				if (block != null) {
					trees.add(buildTree(block));
				}
				block = new HashMap<String, String>();
				continue;
			}
			if (line.equals("end of trees")) {
				if (block != null) {
					trees.add(buildTree(block));
					block = null;
				}
				break;
			}
			if (line.equals("average_output")) {
				averageOutput = true;
				continue;
			}
			int eq = line.indexOf('=');
			if (eq < 0) {
				continue;
			}
			String key = line.substring(0, eq);
			String value = line.substring(eq + 1);
			if (block != null) {
				block.put(key, value);
			} else {
				header.put(key, value);
			}
		}
		if (block != null) {
			trees.add(buildTree(block));
		}
		if (trees.isEmpty()) {
			throw new IOException("not a LightGBM text model: no trees found");
		}

		String names = header.get("feature_names");
		if (names == null) {
			throw new IOException("not a LightGBM text model: feature_names missing");
		}
		String[] featureNames = names.trim().split(" ");
		String numClass = header.get("num_class");
		String perIter = header.get("num_tree_per_iteration");
		if ((numClass != null && !numClass.trim().equals("1")) || (perIter != null && !perIter.trim().equals("1"))) {
			throw new IOException("only single-output regression models are supported (num_class=" + numClass
					+ ", num_tree_per_iteration=" + perIter + ")");
		}
		String objective = header.getOrDefault("objective", "regression").trim();
		Transform transform = objectiveTransform(objective);
		return new LightGBMModel(featureNames, trees.toArray(new Tree[0]), averageOutput, transform, objective);
	}

	private static Transform objectiveTransform(String objective) throws IOException {
		String[] tokens = objective.split("\\s+");
		String name = tokens[0];
		switch (name) {
		case "regression":
		case "regression_l2":
		case "mean_squared_error":
		case "mse":
		case "l2":
		case "regression_l1":
		case "l1":
		case "mean_absolute_error":
		case "mae":
		case "huber":
		case "fair":
		case "quantile":
		case "mape":
			return Arrays.asList(tokens).contains("sqrt") ? Transform.SQUARE : Transform.IDENTITY;
		case "poisson":
		case "gamma":
		case "tweedie":
			return Transform.EXP;
		default:
			throw new IOException("unsupported objective '" + objective + "': only regression models are supported");
		}
	}

	private static Tree buildTree(Map<String, String> b) throws IOException {
		if ("1".equals(b.get("is_linear"))) {
			throw new IOException("linear trees (linear_tree=true) are not supported");
		}
		Tree t = new Tree();
		t.numLeaves = Integer.parseInt(required(b, "num_leaves"));
		t.leafValue = doubles(required(b, "leaf_value"));
		if (t.numLeaves <= 1) {
			return t;
		}
		t.splitFeature = ints(required(b, "split_feature"));
		t.threshold = doubles(required(b, "threshold"));
		t.decisionType = ints(required(b, "decision_type"));
		t.leftChild = ints(required(b, "left_child"));
		t.rightChild = ints(required(b, "right_child"));
		if (b.containsKey("cat_boundaries")) {
			t.catBoundaries = ints(b.get("cat_boundaries"));
			t.catThreshold = ints(required(b, "cat_threshold"));
		}
		return t;
	}

	private static String required(Map<String, String> b, String key) throws IOException {
		String v = b.get(key);
		if (v == null) {
			throw new IOException("tree field '" + key + "' missing");
		}
		return v;
	}

	private static double[] doubles(String s) {
		s = s.trim();
		if (s.isEmpty()) {
			return new double[0];
		}
		String[] p = s.split(" ");
		double[] out = new double[p.length];
		for (int i = 0; i < p.length; i++) {
			out[i] = Double.parseDouble(p[i]);
		}
		return out;
	}

	private static int[] ints(String s) {
		s = s.trim();
		if (s.isEmpty()) {
			return new int[0];
		}
		String[] p = s.split(" ");
		int[] out = new int[p.length];
		for (int i = 0; i < p.length; i++) {
			// cat_threshold words are unsigned 32-bit
			out[i] = (int) Long.parseLong(p[i]);
		}
		return out;
	}
}
