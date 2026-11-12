package predictor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.fail;

import java.io.IOException;

import org.junit.Test;

public class LightGBMModelTest {

	private static String model(String objective, String trees) {
		return "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=1\n"
				+ "objective=" + objective + "\nfeature_names=a b\n\n" + trees + "\nend of trees\n\nparameters:\n";
	}

	private static final String NUMERIC = "Tree=0\nnum_leaves=3\nnum_cat=0\nsplit_feature=0 1\nthreshold=0.5 2\n"
			+ "decision_type=2 2\nleft_child=1 -1\nright_child=-2 -3\nleaf_value=1 2 3\nis_linear=0\n\n"
			+ "Tree=1\nnum_leaves=1\nnum_cat=0\nleaf_value=0.5\nis_linear=0\n\n";

	@Test
	public void sumsTreesAndWalksSplits() throws IOException {
		LightGBMModel m = LightGBMModel.fromString(model("regression", NUMERIC));
		assertArrayEquals(new String[] { "a", "b" }, m.getFeatureNames());
		assertEquals(2, m.getNumTrees());
		assertEquals(1.5, m.predict(new double[] { 0, 1 }), 0);
		assertEquals(3.5, m.predict(new double[] { 0, 5 }), 0);
		assertEquals(2.5, m.predict(new double[] { 1, 0 }), 0);
		assertEquals(1.5, m.predict(new double[] { 0.5, 2 }), 0); // <= goes left
	}

	@Test
	public void missingValues() throws IOException {
		// decision_type 10 = default left, missing type NaN
		String nanLeft = "Tree=0\nnum_leaves=2\nnum_cat=0\nsplit_feature=0\nthreshold=0.5\ndecision_type=10\n"
				+ "left_child=-1\nright_child=-2\nleaf_value=1 2\nis_linear=0\n\n";
		assertEquals(1, LightGBMModel.fromString(model("regression", nanLeft)).predict(new double[] { Double.NaN, 0 }),
				0);
		// decision_type 8 = default right, missing type NaN
		String nanRight = nanLeft.replace("decision_type=10", "decision_type=8");
		assertEquals(2,
				LightGBMModel.fromString(model("regression", nanRight)).predict(new double[] { Double.NaN, 0 }), 0);
		// missing type none: NaN is read as 0
		String none = nanLeft.replace("decision_type=10", "decision_type=0");
		assertEquals(1, LightGBMModel.fromString(model("regression", none)).predict(new double[] { Double.NaN, 0 }),
				0);
	}

	@Test
	public void categoricalSplit() throws IOException {
		// categories {1, 2} go left (bitset 0b110 = 6)
		String cat = "Tree=0\nnum_leaves=2\nnum_cat=1\nsplit_feature=0\nthreshold=0\ndecision_type=1\n"
				+ "left_child=-1\nright_child=-2\nleaf_value=10 20\ncat_boundaries=0 1\ncat_threshold=6\n"
				+ "is_linear=0\n\n";
		LightGBMModel m = LightGBMModel.fromString(model("regression", cat));
		assertEquals(10, m.predict(new double[] { 1, 0 }), 0);
		assertEquals(10, m.predict(new double[] { 2, 0 }), 0);
		assertEquals(20, m.predict(new double[] { 0, 0 }), 0);
		assertEquals(20, m.predict(new double[] { 3, 0 }), 0);
		assertEquals(20, m.predict(new double[] { 99, 0 }), 0);
		assertEquals(20, m.predict(new double[] { -1, 0 }), 0);
	}

	@Test
	public void objectiveTransforms() throws IOException {
		assertEquals(Math.exp(1.5),
				LightGBMModel.fromString(model("poisson", NUMERIC)).predict(new double[] { 0, 1 }), 1e-12);
		assertEquals(2.25,
				LightGBMModel.fromString(model("regression sqrt", NUMERIC)).predict(new double[] { 0, 1 }), 1e-12);
	}

	@Test
	public void rejectsModelsItCannotScore() {
		for (String bad : new String[] { model("binary sigmoid:1", NUMERIC),
				model("regression", NUMERIC).replace("num_class=1", "num_class=3"),
				model("regression", NUMERIC.replace("is_linear=0", "is_linear=1")) }) {
			try {
				LightGBMModel.fromString(bad);
				fail("accepted unsupported model");
			} catch (IOException expected) {
				// ok
			}
		}
	}

	@Test
	public void unknownFeatureIsNamed() {
		try {
			FeatureRegistry.resolve(new String[] { "log_est", "no_such_feature" });
			fail();
		} catch (IllegalArgumentException e) {
			assertEquals(true, e.getMessage().contains("no_such_feature"));
		}
	}
}
