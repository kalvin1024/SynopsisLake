"""Learned merge policy: which boundaries to drop when ARM reshapes the aligned histogram.

ARM (Align-Reshape-Merge) starts from the aligned histogram whose boundaries are the union of the source
histograms' boundaries, and drops boundaries until the target resolution is reached. Its reshape step scores
candidate drops with a hand-written density-difference objective. This module learns that score instead.

    candidate   one interior boundary on one axis; dropping it merges the two adjacent buckets
    features    describe the two buckets and the boundary (FEATURES below); no query workload is needed to
                use the policy, only to train it
    label       log of the increase in mean relative error, on training queries with known counts, when that
                boundary is dropped in the current state (marginal damage)
    policy      repeatedly drop the candidate with the lowest predicted damage, alternating to the axis that
                still has the most drops to make, until the target resolution

The Java side (predictor.MergePolicy) computes the same features and runs the same loop, so the model.txt this
module trains can be used by ErrorPredictionLogExp -mergePolicy. `golden` writes a fixture that the Java test
checks the features against.

    python ml/merge_policy.py train  RUN_DIR DATASET_DIR OUT_DIR [--synthetic-data DIR] [--seed 0]
    python ml/merge_policy.py eval   RUN_DIR DATASET_DIR MODEL   [--seed 0]
    python ml/merge_policy.py select RUN_DIR DATASET_DIR OUT_DIR [--model M] [--tolerance 0.02] [--seed 0]
    python ml/merge_policy.py golden OUT_FILE

RUN_DIR is a run folder from ErrorPredictionLogExp (arm_<res>.txt, src_<i>.txt, rows.csv); DATASET_DIR holds
queries-Ans.txt (minLon,maxLon,minLat,maxLat,count per line). The aligned histogram is the arm_ file with the
largest resolution.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FEATURES = ["axis", "progress", "pos", "log_wl", "log_wr", "log_w_ratio", "log_cl", "log_cr", "log_cm", "share",
            "dens_gap", "tv", "tv_w", "src_n", "log_dens_m"]
MIN_EXACT = 10          # queries with fewer true points are not used (as in the error predictor)
MIN_DAMAGE = 1e-5       # labels are log(max(damage, MIN_DAMAGE)): only the ranking of candidates matters


# ---------------------------------------------------------------------------------------------------------
# histograms (same file format as ErrorPredictionLogExp: rate / lon boundaries / lat boundaries / count rows)
# ---------------------------------------------------------------------------------------------------------

def load_hist(path):
    lines = open(path).read().strip().split("\n")
    lon = np.array(lines[1].split(","), float)
    lat = np.array(lines[2].split(","), float)
    counts = np.array([r.split(",") for r in lines[3:]], float)
    return float(lines[0]), lon, lat, counts


def overlap_frac(bnd, lo, hi):
    """Fraction of each bucket [bnd_i, bnd_i+1] covered by the interval [lo, hi]; (Q,) intervals -> (Q, n)."""
    left = np.maximum(bnd[None, :-1], lo[:, None])
    right = np.minimum(bnd[None, 1:], hi[:, None])
    return np.clip(right - left, 0, None) / (bnd[1:] - bnd[:-1])[None, :]


def answers(lon, lat, counts, q):
    """Range-count estimates (uniform inside buckets, as SimpleSpatialHistogramOpt.RangeQuery); q: (Q,4)."""
    return np.einsum("qi,ij,qj->q", overlap_frac(lon, q[:, 0], q[:, 1]), counts, overlap_frac(lat, q[:, 2], q[:, 3]))


def merge_to(lon, lat, counts, keep_lon, keep_lat):
    """Histogram on the kept boundaries (boolean masks over the canonical boundaries): sums the merged cells."""
    li, la = np.where(keep_lon)[0], np.where(keep_lat)[0]
    merged = np.add.reduceat(np.add.reduceat(counts, li[:-1], axis=0), la[:-1], axis=1)
    return lon[li], lat[la], merged


def source_coincidence(canon, source_bounds, tol=1e-9):
    """For each canonical boundary, how many source histograms have a boundary there."""
    return np.array([sum(np.abs(b - x).min() < tol for b in source_bounds) for x in canon], float)


# ---------------------------------------------------------------------------------------------------------
# candidates: features and true marginal damage
# ---------------------------------------------------------------------------------------------------------

def candidate_features(m, bnd, axis, n0, src_n):
    """Features of every candidate drop on one axis.

    m: counts with this axis first (lon axis: counts; lat axis: counts.T); bnd: this axis's current boundaries
    (n+1 values); n0: this axis's resolution before any drop; src_n: per interior boundary (n-1 values), the
    number of sources with a boundary there. Candidate j drops bnd[j+1], merging buckets j and j+1.
    Returns (n-1, len(FEATURES)) in FEATURES order.
    """
    n = len(bnd) - 1
    w = np.diff(bnd)
    rowc = m.sum(1)
    total = m.sum()
    cl, cr, wl, wr = rowc[:-1], rowc[1:], w[:-1], w[1:]
    dl, dr = cl / wl, cr / wr
    pl = m[:-1] / (cl[:, None] + 1e-9)
    pr = m[1:] / (cr[:, None] + 1e-9)
    tv = 0.5 * np.abs(pl - pr).sum(1)
    j = np.arange(n - 1)
    cols = [np.full(n - 1, float(axis)), np.full(n - 1, n / n0), j / n, np.log(wl), np.log(wr), np.log(wl / wr),
            np.log1p(cl), np.log1p(cr), np.log1p(cl + cr), (cl + cr) / total,
            np.abs(np.log1p(dl) - np.log1p(dr)), tv, tv * np.log1p(np.minimum(cl, cr)), np.asarray(src_n, float),
            np.log1p((cl + cr) / (wl + wr))]
    return np.column_stack(cols)


def marginal_damage(m, bnd, other, axis_queries, other_queries, gt):
    """True increase in mean relative error (over the training queries) for dropping each interior boundary.

    axis_queries / other_queries: (Q,2) query [lo, hi] on this axis / the other axis."""
    fx = overlap_frac(bnd, axis_queries[:, 0], axis_queries[:, 1])
    fy = overlap_frac(other, other_queries[:, 0], other_queries[:, 1])
    g = fy @ m.T                                   # (Q, n): per query, the count each bucket row contributes
    a0 = (fx * g).sum(1)
    w = np.diff(bnd)
    ov = fx * w[None, :]
    merged_frac = (ov[:, :-1] + ov[:, 1:]) / (w[:-1] + w[1:])[None, :]
    a1 = a0[:, None] + merged_frac * (g[:, :-1] + g[:, 1:]) - fx[:, :-1] * g[:, :-1] - fx[:, 1:] * g[:, 1:]
    err0 = np.abs(a0 - gt) / gt
    err1 = np.abs(a1 - gt[:, None]) / gt[:, None]
    return (err1 - err0[:, None]).mean(0)


class Aligned:
    """The aligned (canonical) histogram plus what the policy needs to know about the sources."""

    def __init__(self, lon, lat, counts, src_bounds_lon, src_bounds_lat):
        self.lon, self.lat, self.counts = lon, lat, counts
        self.n0 = (len(lon) - 1, len(lat) - 1)
        self.src_n = (source_coincidence(lon, src_bounds_lon), source_coincidence(lat, src_bounds_lat))

    @classmethod
    def from_run_dir(cls, run_dir):
        res = sorted(int(f[4:-4]) for f in os.listdir(run_dir) if f.startswith("arm_"))
        _, lon, lat, counts = load_hist(os.path.join(run_dir, f"arm_{res[-1]}.txt"))
        srcs = [load_hist(os.path.join(run_dir, f)) for f in sorted(os.listdir(run_dir)) if f.startswith("src_")]
        return cls(lon, lat, counts, [s[1] for s in srcs], [s[2] for s in srcs]), res

    def state(self, keep_lon, keep_lat):
        return merge_to(self.lon, self.lat, self.counts, keep_lon, keep_lat)

    def candidates(self, axis, keep_lon, keep_lat, labels=None):
        """(features, canonical index of each candidate boundary, damage or None) for one axis."""
        lon, lat, counts = self.state(keep_lon, keep_lat)
        keep = keep_lon if axis == 0 else keep_lat
        idx = np.where(keep)[0]
        m, bnd, other = (counts, lon, lat) if axis == 0 else (counts.T, lat, lon)
        interior = idx[1:-1]
        f = candidate_features(m, bnd, axis, self.n0[axis], self.src_n[axis][interior])
        if labels is None:
            return f, interior, None
        q, gt = labels
        a, o = (q[:, 0:2], q[:, 2:4]) if axis == 0 else (q[:, 2:4], q[:, 0:2])
        return f, interior, marginal_damage(m, bnd, other, a, o, gt)


def trajectory_snapshots(aligned, targets, pick, collect=None):
    """One greedy run down to the smallest target, keeping the histogram at every target resolution (square
    targets, lon == lat). The first target must equal the aligned lon resolution."""
    keep = [np.ones(len(aligned.lon), bool), np.ones(len(aligned.lat), bool)]
    snaps, samples = {}, []
    final = min(targets)
    step = 0
    while True:
        n = [keep[a].sum() - 1 for a in (0, 1)]
        if n[0] == n[1] and n[0] in targets:
            snaps[n[0]] = aligned.state(*keep)
        if max(n) <= final:
            break
        axis = 0 if n[0] >= n[1] else 1
        f, interior, dmg = aligned.candidates(axis, keep[0], keep[1], collect)
        if collect is not None:
            samples.append((step, f, dmg))
        keep[axis][interior[pick(f, axis, dmg)]] = False
        step += 1
    return snaps, samples


# ---------------------------------------------------------------------------------------------------------
# training and evaluation
# ---------------------------------------------------------------------------------------------------------

def fit_model(x, y, seed, step_ids):
    """LightGBM when available (its model.txt is what Java reads); scikit-learn otherwise (not exportable)."""
    held = step_ids % 5 == 0
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.08, num_leaves=31, min_child_samples=40,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1,
                                  random_state=seed)
        model.fit(x[~held], y[~held], eval_set=[(x[held], y[held])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        return model, True
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=seed).fit(
            x[~held], y[~held]), False


def load_queries(dataset_dir, seed, train_n=4000):
    """Train / test queries (count >= MIN_EXACT) split by query id with the same 50/20/30 split as the error
    predictor. Returns (train (q, gt), test (q, gt), test qIds)."""
    from gate_eval import split_ids
    q = pd.read_csv(os.path.join(dataset_dir, "queries-Ans.txt"), header=None).to_numpy()
    ok = np.where(q[:, 4] >= MIN_EXACT)[0]
    tr, _, te = split_ids(ok, seed)
    rng = np.random.default_rng(seed)
    tr = rng.choice(np.array(sorted(tr)), min(train_n, len(tr)), replace=False)
    te = np.array(sorted(te))
    return (q[tr, :4], q[tr, 4]), (q[te, :4], q[te, 4]), te


def synthetic_queries(points, n, rng, width, height, min_count=MIN_EXACT):
    """Workload-agnostic training queries: centres uniform over the data's bounding box (not data-driven), sizes
    log-uniform in width = (lo, hi) and height = (lo, hi) degrees, exact counts from `points` (N,2). Queries
    with fewer than min_count points are dropped."""
    p = points[np.argsort(points[:, 0], kind="stable")]
    x, y = p[:, 0], p[:, 1]
    w = np.exp(rng.uniform(np.log(width[0]), np.log(width[1]), n))
    h = np.exp(rng.uniform(np.log(height[0]), np.log(height[1]), n))
    cx = rng.uniform(x.min(), x.max(), n)
    cy = rng.uniform(y.min(), y.max(), n)
    q = np.column_stack([cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2])
    a, b = np.searchsorted(x, q[:, 0], "left"), np.searchsorted(x, q[:, 1], "right")
    gt = np.array([((y[i:j] >= r[2]) & (y[i:j] <= r[3])).sum() for r, i, j in zip(q, a, b)], float)
    keep = gt >= min_count
    return q[keep], gt[keep]


def oracle_pick(f, axis, damage):
    return int(np.argmin(damage))


def train(aligned, train_labels, seed, resolutions):
    """Collect (features, damage) along the oracle-greedy trajectory on the training queries and fit a model."""
    _, samples = trajectory_snapshots(aligned, set(resolutions), oracle_pick, collect=train_labels)
    x = pd.DataFrame(np.vstack([s[1] for s in samples]), columns=FEATURES)   # named: Java maps columns by name
    y = np.log(np.maximum(np.concatenate([s[2] for s in samples]), MIN_DAMAGE))
    steps = np.concatenate([np.full(len(s[1]), s[0]) for s in samples])
    model, is_lgb = fit_model(x, y, seed, steps)
    return model, is_lgb, len(x)


def mean_rel_err(hist, q, gt):
    lon, lat, counts = hist
    return float((np.abs(answers(lon, lat, counts, q) - gt) / gt).mean())


def evaluate(aligned, resolutions, model, test, run_dir, test_ids):
    """Mean relative error on the test queries at every resolution: ARM (its logged rows.csv), the learned
    policy, and uniformly spaced boundaries."""
    q, gt = test
    snaps, _ = trajectory_snapshots(aligned, set(resolutions), lambda f, a, d: int(np.argmin(model.predict(pd.DataFrame(f, columns=FEATURES)))))
    rows = pd.read_csv(os.path.join(run_dir, "rows.csv"), usecols=["lonRes", "qId", "relErr", "gt"])
    rows = rows[rows.qId.isin(test_ids) & (rows["gt"] >= MIN_EXACT)]   # rows.gt is DataFrame.gt, not the column
    arm = rows.groupby("lonRes").relErr.mean()
    out = []
    for res in sorted(resolutions, reverse=True):
        ku = [np.zeros(len(aligned.lon), bool), np.zeros(len(aligned.lat), bool)]
        for k, size in zip(ku, (len(aligned.lon), len(aligned.lat))):
            k[np.round(np.linspace(0, size - 1, res + 1)).astype(int)] = True
        out.append({"res": res, "arm": float(arm.get(res, np.nan)), "learned": mean_rel_err(snaps[res], q, gt),
                    "uniform": mean_rel_err(aligned.state(*ku), q, gt)})
    return pd.DataFrame(out).set_index("res")


# ---------------------------------------------------------------------------------------------------------
# guardrail: choose ARM, the mass rule or the learned policy per dataset
# ---------------------------------------------------------------------------------------------------------

STRATEGIES = ("arm", "mass", "learned")
WINDOW = 0.73           # only resolutions at or below this fraction of the full resolution are compared


def load_split(dataset_dir, seed, which):
    """Queries (count >= MIN_EXACT) of the 'val' (20%) or 'test' (30%) split by query id, as in load_queries."""
    from gate_eval import split_ids
    q = pd.read_csv(os.path.join(dataset_dir, "queries-Ans.txt"), header=None).to_numpy()
    ok = np.where(q[:, 4] >= MIN_EXACT)[0]
    _, va, te = split_ids(ok, seed)
    ids = np.array(sorted(va if which == "val" else te))
    return q[ids, :4], q[ids, 4]


def mass_pick(f, axis, damage):
    """The simple rule: merge the pair of buckets with the fewest points."""
    return int(np.argmin(f[:, FEATURES.index("log_cm")]))


def per_query_errors(aligned, resolutions, run_dir, queries, booster=None):
    """{strategy: {res: relative error per query}} for ARM (its arm_<res>.txt), the mass rule and, with a model,
    the learned policy."""
    q, gt = queries
    out = {"arm": {}, "mass": {}}
    for r in resolutions:
        _, lon, lat, counts = load_hist(os.path.join(run_dir, f"arm_{r}.txt"))
        out["arm"][r] = np.abs(answers(lon, lat, counts, q) - gt) / gt
    snaps, _ = trajectory_snapshots(aligned, set(resolutions), mass_pick)
    out["mass"] = {r: np.abs(answers(*snaps[r][:3], q) - gt) / gt for r in resolutions}
    if booster is not None:
        pick = lambda f, a, d: int(np.argmin(booster.predict(pd.DataFrame(f, columns=FEATURES))))
        snaps, _ = trajectory_snapshots(aligned, set(resolutions), pick)
        out["learned"] = {r: np.abs(answers(*snaps[r][:3], q) - gt) / gt for r in resolutions}
    return out


def summarize(errors, queries, full_res):
    """Per strategy: mean error over the window's resolutions, overall and within each query-size third (by area).
    Returns a DataFrame indexed by strategy with columns overall, small, medium, large."""
    q, _ = queries
    area = (q[:, 1] - q[:, 0]) * (q[:, 3] - q[:, 2])
    lo, hi = np.quantile(area, [1 / 3, 2 / 3])
    groups = {"overall": np.ones(len(q), bool), "small": area <= lo,
              "medium": (area > lo) & (area <= hi), "large": area > hi}
    rows = {}
    for name, per_res in errors.items():
        res = [r for r in per_res if r <= WINDOW * full_res]
        rows[name] = {g: float(np.mean([per_res[r][m].mean() for r in res])) for g, m in groups.items()}
    return pd.DataFrame(rows).T


def choose(summary, tolerance):
    """Lowest overall error among the strategies that are no worse than ARM by more than `tolerance` (relative)
    in every query-size group. ARM is always eligible."""
    arm = summary.loc["arm"]
    eligible = {}
    for name in summary.index:
        ratio = summary.loc[name] / arm
        if name == "arm" or ratio.drop("overall").max() <= 1 + tolerance:
            eligible[name] = summary.loc[name, "overall"]
        else:
            worst = ratio.drop("overall").idxmax()
            print(f"  not eligible - {name}: {ratio[worst] - 1:+.1%} vs ARM on {worst} queries")
    best = min(eligible, key=eligible.get)
    return best, eligible


def select_strategy(aligned, resolutions, run_dir, dataset_dir, out_dir, model_path, seed, tolerance):
    """Pick the strategy on validation queries, confirm on test queries, write merge_strategy.properties and
    merge_strategy_report.csv into out_dir."""
    booster = None
    if model_path and os.path.exists(model_path):
        try:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=model_path)
        except ImportError:
            print("lightgbm is not installed: the learned policy is not considered")
    val, test = load_split(dataset_dir, seed, "val"), load_split(dataset_dir, seed, "test")
    full = aligned.n0[0]
    v = summarize(per_query_errors(aligned, resolutions, run_dir, val, booster), val, full)
    chosen, eligible = choose(v, tolerance)
    t = summarize(per_query_errors(aligned, resolutions, run_dir, test, booster), test, full)
    os.makedirs(out_dir, exist_ok=True)
    report = pd.concat({"validation": v, "test": t}, names=["split", "strategy"])
    report["chosen"] = [s == chosen for _, s in report.index]
    report.round(5).to_csv(os.path.join(out_dir, "merge_strategy_report.csv"))
    with open(os.path.join(out_dir, "merge_strategy.properties"), "w") as w:
        w.write(f"strategy={chosen}\n")
        if chosen == "learned":
            w.write(f"model={os.path.relpath(model_path, out_dir)}\n")
    return chosen, v, t


# ---------------------------------------------------------------------------------------------------------
# golden fixture for the Java parity test
# ---------------------------------------------------------------------------------------------------------

def _csv(values):
    return ",".join(repr(float(v)) for v in values)


def write_golden(path):
    """A small fixed histogram and the features of every candidate on both axes, for predictor.MergePolicyTest."""
    rng = np.random.default_rng(5)
    lon = np.array([0.0, 0.5, 1.2, 2.0, 2.4, 3.5, 4.0, 5.1, 6.0, 7.5, 8.0, 9.0, 10.0])
    lat = np.array([-3.0, -2.2, -1.0, 0.1, 0.9, 2.0, 2.5, 4.0, 5.0, 6.5, 7.0])
    counts = np.round(rng.gamma(0.6, 40.0, (len(lon) - 1, len(lat) - 1)))
    src_lon = np.array([1, 2, 1, 1, 3, 1, 2, 1, 1, 2, 1], float)           # per interior boundary
    src_lat = np.array([1, 1, 2, 1, 1, 3, 1, 2, 1], float)
    aligned = Aligned.__new__(Aligned)
    aligned.lon, aligned.lat, aligned.counts = lon, lat, counts
    aligned.n0 = (len(lon) - 1, len(lat) - 1)
    aligned.src_n = (np.r_[0, src_lon, 0], np.r_[0, src_lat, 0])
    keep = [np.ones(len(lon), bool), np.ones(len(lat), bool)]
    keep[0][[3, 7]] = False                      # a non-trivial state: some boundaries already dropped
    keep[1][[4]] = False
    with open(path, "w") as w:
        w.write("lon," + _csv(lon) + "\n")
        w.write("lat," + _csv(lat) + "\n")
        for row in counts:
            w.write("counts," + _csv(row) + "\n")
        w.write("src_lon," + _csv(src_lon) + "\n")
        w.write("src_lat," + _csv(src_lat) + "\n")
        w.write("keep_lon," + ",".join("1" if k else "0" for k in keep[0]) + "\n")
        w.write("keep_lat," + ",".join("1" if k else "0" for k in keep[1]) + "\n")
        w.write("names," + ",".join(FEATURES) + "\n")
        for axis in (0, 1):
            f, interior, _ = aligned.candidates(axis, keep[0], keep[1])
            for k, row in enumerate(f):
                w.write(f"feat,{axis},{interior[k]}," + _csv(row) + "\n")


# ---------------------------------------------------------------------------------------------------------

def _resolutions(run_dir):
    return sorted(int(f[4:-4]) for f in os.listdir(run_dir) if f.startswith("arm_"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("run_dir"), t.add_argument("dataset_dir"), t.add_argument("out_dir")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--train-queries", type=int, default=4000)
    t.add_argument("--synthetic-data", help="folder with 0.csv.. (tab-separated lon lat): train on uniformly "
                   "placed synthetic queries with exact counts from these points instead of the dataset queries")
    t.add_argument("--synthetic-width", type=float, nargs=2, default=[0.5, 12.0])
    t.add_argument("--synthetic-height", type=float, nargs=2, default=[0.25, 6.0])
    e = sub.add_parser("eval")
    e.add_argument("run_dir"), e.add_argument("dataset_dir"), e.add_argument("model")
    e.add_argument("--seed", type=int, default=0)
    sl = sub.add_parser("select", help="guardrail: choose ARM, the mass rule or the learned policy on validation queries")
    sl.add_argument("run_dir"), sl.add_argument("dataset_dir"), sl.add_argument("out_dir")
    sl.add_argument("--model", help="merge_policy.txt (default OUT_DIR/merge_policy.txt); without one only ARM and the mass rule compete")
    sl.add_argument("--tolerance", type=float, default=0.02,
                    help="largest relative error increase over ARM allowed in any query-size group (default 0.02)")
    sl.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("golden")
    g.add_argument("out_file")
    a = ap.parse_args()

    if a.cmd == "golden":
        write_golden(a.out_file)
        print("wrote", a.out_file)
        return
    aligned, res = Aligned.from_run_dir(a.run_dir)
    if a.cmd == "select":
        model = a.model or os.path.join(a.out_dir, "merge_policy.txt")
        chosen, v, t = select_strategy(aligned, res, a.run_dir, a.dataset_dir, a.out_dir, model, a.seed, a.tolerance)
        print("validation (mean relative error over resolutions <= 73% of full, by query size):")
        print(v.round(4).to_string())
        print("held-out test:")
        print(t.round(4).to_string())
        print(f"chosen: {chosen} -> {a.out_dir}/merge_strategy.properties")
        return
    train_labels, test, test_ids = load_queries(a.dataset_dir, a.seed, getattr(a, "train_queries", 4000))
    if a.cmd == "train":
        if a.synthetic_data:
            pts = np.vstack([pd.read_csv(os.path.join(a.synthetic_data, f"{i}.csv"), sep="\t", header=None,
                                         usecols=[0, 1]).to_numpy()
                             for i in range(len([f for f in os.listdir(a.synthetic_data) if f.endswith(".csv")]))])
            q, gt = synthetic_queries(pts, 6 * a.train_queries, np.random.default_rng(a.seed),
                                      a.synthetic_width, a.synthetic_height)
            k = np.random.default_rng(a.seed).permutation(len(q))[:a.train_queries]
            train_labels = (q[k], gt[k])
            print(f"training on {len(k)} synthetic uniform-location queries")
        model, is_lgb, rows = train(aligned, train_labels, a.seed, res)
        os.makedirs(a.out_dir, exist_ok=True)
        if is_lgb:
            model.booster_.save_model(os.path.join(a.out_dir, "merge_policy.txt"))
            print(f"trained on {rows} candidate rows; wrote {a.out_dir}/merge_policy.txt")
        else:
            print("lightgbm is not installed: trained a scikit-learn model for evaluation only (no model file)")
        print(evaluate(aligned, res, model, test, a.run_dir, test_ids).round(4).to_string())
    else:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=a.model)
        df = evaluate(aligned, res, booster, test, a.run_dir, test_ids)
        print(df.round(4).to_string())
        small = df[df.index <= 0.73 * aligned.n0[0]]
        print(f"\nmean over resolutions <= 73% of full: ARM {small.arm.mean():.4f}, learned {small.learned.mean():.4f} "
              f"({small.learned.mean() / small.arm.mean() - 1:+.1%}), uniform {small.uniform.mean():.4f}")


if __name__ == "__main__":
    main()
