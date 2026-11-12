package predictor;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * How a dataset's synopses are reshaped, as chosen by the guardrail (`python ml/merge_policy.py select`):
 * ARM (null policy), the simple mass rule, or the learned merge policy. Read from merge_strategy.properties:
 *
 * <pre>
 * strategy=arm|mass|learned
 * model=merge_policy.txt      (learned only; relative to the properties file)
 * </pre>
 */
public class MergeStrategy {

	public final String name;
	/** null when the strategy is ARM. */
	public final MergePolicy policy;

	private MergeStrategy(String name, MergePolicy policy) {
		this.name = name;
		this.policy = policy;
	}

	public static MergeStrategy arm() {
		return new MergeStrategy("arm", null);
	}

	public static MergeStrategy load(String propertiesPath) throws IOException {
		Properties p = new Properties();
		try (InputStream in = new FileInputStream(propertiesPath)) {
			p.load(in);
		}
		String name = p.getProperty("strategy", "").trim();
		if (name.equals("arm")) {
			return arm();
		} else if (name.equals("mass")) {
			return new MergeStrategy(name, MergePolicy.massRule());
		} else if (name.equals("learned")) {
			String model = p.getProperty("model");
			if (model == null || model.trim().isEmpty()) {
				throw new IllegalArgumentException(propertiesPath + ": strategy=learned needs model=");
			}
			File f = new File(model.trim());
			if (!f.isAbsolute()) {
				f = new File(new File(propertiesPath).getAbsoluteFile().getParentFile(), model.trim());
			}
			return new MergeStrategy(name, MergePolicy.load(f.getPath()));
		}
		throw new IllegalArgumentException(propertiesPath + ": strategy must be arm, mass or learned, got '" + name + "'");
	}
}
