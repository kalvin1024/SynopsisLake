package predictor;

import java.util.ArrayList;
import java.util.List;

/**
 * Chooses how far to merge: runs the gate over a workload with known exact counts at every merge rate in a
 * {@link SynopsisStore} and picks the smallest synopsis (lowest merge rate) whose observed violation stays within
 * the accuracy target.
 *
 * A query violates when it is answered from the synopsis and its relative error is above eps. Coverage is the
 * share of queries answered from the synopsis; the rest would be read from the raw data. A rate qualifies when
 * its violation is at most delta and its coverage is at least minCoverage (a synopsis that answers almost
 * nothing saves nothing). If no rate qualifies the plan has no chosen rate and the raw data is read.
 *
 * Use a workload that was not used to calibrate the cutoffs, otherwise the violation is optimistic.
 */
public class MergeRatePlanner {

	public static final class RateResult {
		public final double mergeRate;
		public final double cutoff;
		/** queries evaluated (exact count at least minExact) */
		public final int queries;
		public final int answered;
		public final double coverage;
		/** share of answered queries with relative error above eps */
		public final double violation;
		public final double meanRelErrAnswered;
		public final boolean meetsTarget;

		RateResult(double mergeRate, double cutoff, int queries, int answered, double violation, double meanRelErr,
				boolean meetsTarget) {
			this.mergeRate = mergeRate;
			this.cutoff = cutoff;
			this.queries = queries;
			this.answered = answered;
			this.coverage = queries == 0 ? 0 : answered / (double) queries;
			this.violation = violation;
			this.meanRelErrAnswered = meanRelErr;
			this.meetsTarget = meetsTarget;
		}
	}

	public static final class Plan {
		/** One row per merge rate, largest first. */
		public final List<RateResult> rates;
		/** The chosen merge rate, or null when none meets the target. */
		public final Double chosenMergeRate;

		Plan(List<RateResult> rates, Double chosen) {
			this.rates = rates;
			this.chosenMergeRate = chosen;
		}
	}

	private final double eps;
	private final double delta;
	private final double minCoverage;
	private final double minExact;

	/**
	 * @param eps         error limit
	 * @param delta       allowed share of synopsis answers above eps
	 * @param minCoverage smallest share of queries the synopsis must answer for a rate to qualify
	 * @param minExact    queries with a smaller exact count are skipped (relative error is not meaningful; the
	 *                    training notebooks drop them too)
	 */
	public MergeRatePlanner(double eps, double delta, double minCoverage, double minExact) {
		this.eps = eps;
		this.delta = delta;
		this.minCoverage = minCoverage;
		this.minExact = minExact;
	}

	/** @param queries as returned by {@link QueryFile#load}; every query needs its exact count */
	public Plan plan(SynopsisStore store, ErrorPredictor predictor, CutoffTable cutoffs, List<double[]> queries) {
		List<RateResult> results = new ArrayList<RateResult>();
		Double chosen = null;
		for (double rate : store.mergeRates()) {
			double cutoff = cutoffs.forMergeRate(rate);
			GatedRangeQuery gate = new GatedRangeQuery(store.context(rate, predictor.needsBeta()), predictor, cutoff,
					null);
			int n = 0, answered = 0, violations = 0;
			double errSum = 0;
			for (double[] q : queries) {
				double exact = q[4];
				if (Double.isNaN(exact)) {
					throw new IllegalArgumentException("planning needs the exact count of every query");
				}
				if (exact < minExact) {
					continue;
				}
				n++;
				ApproxAnswer a = gate.answer(q[0], q[1], q[2], q[3]);
				if (a.fromSynopsis) {
					answered++;
					double err = Math.abs(exact - a.answer) / exact;
					errSum += err;
					if (err > eps) {
						violations++;
					}
				}
			}
			double violation = answered == 0 ? 0 : violations / (double) answered;
			boolean ok = n > 0 && answered > 0 && answered / (double) n >= minCoverage && violation <= delta;
			results.add(new RateResult(rate, cutoff, n, answered, violation, answered == 0 ? 0 : errSum / answered, ok));
			if (ok && (chosen == null || rate < chosen)) {
				chosen = rate;
			}
		}
		return new Plan(results, chosen);
	}
}
