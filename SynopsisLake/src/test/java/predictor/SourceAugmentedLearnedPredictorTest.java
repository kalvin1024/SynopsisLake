package predictor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;

import org.junit.Test;

import histogram.SimpleSpatialHistogramOpt;

public class SourceAugmentedLearnedPredictorTest {
    private static SourceShapePredictor source() throws Exception {
        Path dir = Paths.get(SourceAugmentedLearnedPredictorTest.class.getResource("/source_shape").toURI());
        return SourceShapePredictor.load(dir.toString(), dir.resolve("children.bin").toString());
    }

    private static LightGBMModel model(String names, int feature) throws Exception {
        return LightGBMModel.fromString("tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\n"
            + "max_feature_idx=2\nobjective=regression\nfeature_names=" + names + "\n\nTree=0\n"
            + "num_leaves=2\nnum_cat=0\nsplit_feature=" + feature + "\nthreshold=0.3\ndecision_type=2\n"
            + "left_child=-1\nright_child=-2\nleaf_value=0.05 0.4\nis_linear=0\n\nend of trees\n");
    }

    private static SynopsisContext context() {
        SimpleSpatialHistogramOpt h = new SimpleSpatialHistogramOpt(0, 0, 2, 2, 1, 1);
        h.getData()[0][0] = 17;
        return new SynopsisContext(h, new SimpleSpatialHistogramOpt[] {h}, 1, null);
    }

    @Test
    public void criticInputsAndExistingRegistryInputsKeepModelOrder() throws Exception {
        Properties properties = new Properties();
        properties.setProperty("label.transform", "identity");
        SourceAugmentedLearnedPredictor p = new SourceAugmentedLearnedPredictor(
            model("critic_log_reference critic_proxy_score log_est", 1), properties, source());
        QueryErrorEstimator bound = p.bind(context());
        assertEquals(.05, bound.estimate(0, 2, 0, 2), 0); // H=R=17, proxy=0.
        assertEquals(.4, bound.estimate(0, .5, 0, .5), 0); // H=1.0625, R=3, proxy>.3.
        assertTrue(p.scoreIsRelativeError());
    }

    @Test
    public void logLabelsAndPropertiesCutoffUseExistingLearnedConventions() throws Exception {
        Properties properties = new Properties();
        properties.setProperty("label.transform", "log");
        properties.setProperty("gate.cutoff", ".2");
        SourceAugmentedLearnedPredictor p = new SourceAugmentedLearnedPredictor(
            model("critic_log_reference critic_proxy_score log_est", 1), properties, source());
        assertEquals(Math.exp(.05), p.bind(context()).estimate(0, 2, 0, 2), 0);
        assertEquals(.2, PredictorFactory.cutoffs(null, null, .25, .05, p).forMergeRate(1), 0);
    }

    @Test
    public void fineReferenceModelCannotDeclareItselfASourceShapeModel() throws Exception {
        Properties properties = new Properties();
        properties.setProperty("critic.reference", "fine");
        try {
            new SourceAugmentedLearnedPredictor(model("critic_log_reference critic_proxy_score log_est", 1),
                                                properties, source());
            fail("different reference semantics were silently accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("different critic.reference"));
        }
    }
}
