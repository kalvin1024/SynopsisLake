"""Model v1 inputs: the v0 inputs (features.py) plus source-level and merge-shift inputs.

Source-level inputs describe the query on each per-file source histogram (`src_<i>.txt`). SynopsisLake keeps one
synopsis per file and merges them at query time, so these are available without reading raw data. They do not
depend on the merge rate, so they are computed once per query and joined to every merge rate.

Merge-shift inputs compare the merged answer with the no-merge answer (`srcSumEst`: the sum of the source
histograms' answers). Both come from synopses only.

    from features_v1 import FEATURES_V1, GROUPS, build_table_v1
    df = build_table_v1(run_dir, dataset_dir)
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from features import E, FEATURES, _query_features, build_table, load_hist

SOURCE_FEATURES = ["src_log_est", "src_bound_rel", "src_edge_share", "src_edge_contrast", "src_in_out",
                   "src_empty_partial", "src_n_partial"]
SHIFT_FEATURES = ["merge_shift", "abs_merge_shift"]
FEATURES_V1 = FEATURES + SOURCE_FEATURES + SHIFT_FEATURES

# Input groups used by the ablations in notebook 06.
GROUPS = {
    "shape": ["area_cells", "n_cells", "n_partial", "side_frac"],
    "edge": ["log_est", "log_full", "bound_rel", "edge_share", "edge_contrast", "in_out", "empty_partial"],
    "scale_free": ["est_frac", "density_ratio"],
    "merge": ["merge_rate", "n_src_overlap", "src_res_min", "src_res_max"],
    "skew": ["skew_pred", "log_alpha"],
    "source": SOURCE_FEATURES,
    "shift": SHIFT_FEATURES,
}


def _hist_context(path):
    """Everything _query_features needs about one histogram, plus its extent."""
    _, bx, by, C = load_hist(path)
    area = np.outer(np.diff(bx), np.diff(by))
    D = C / area
    LC = np.abs(np.log((D + E) / (uniform_filter(D, size=3, mode="nearest") + E)))
    nonempty = D[C > 0]
    med = float(np.median(nonempty)) if nonempty.size else 0.0
    return dict(bx=bx, by=by, C=C, area=area, LC=LC, med=med, total=float(C.sum()),
                extent=(bx[0], bx[-1], by[0], by[-1]))


def source_features(run_dir, dataset_dir):
    """One row per query: the query described on every source histogram it overlaps, summed or
    weighted by each source's share of the no-merge estimate."""
    queries = pd.read_csv(os.path.join(dataset_dir, "queries-Ans.txt"), header=None).to_numpy()
    sources = [_hist_context(p) for p in sorted(glob.glob(os.path.join(run_dir, "src_*.txt")))]
    out = []
    for qid, q in enumerate(queries):
        a, b, c, d = q[:4]
        est = lo = width = contrast = in_out = empty = n_partial = 0.0
        for s in sources:
            x0, x1, y0, y1 = s["extent"]
            if not (x0 < b and a < x1 and y0 < d and c < y1):
                continue
            f = _query_features((a, b, c, d), s["bx"], s["by"], s["C"], s["area"], s["LC"], s["med"], s["total"], [])
            e = f["est_py"]
            est += e
            lo += np.expm1(f["log_full"])                      # count in fully covered cells
            width += f["bound_rel"] * max(e, 1.0)              # max possible minus min possible count
            contrast += e * f["edge_contrast"]
            in_out += e * f["in_out"]
            empty += e * f["empty_partial"]
            n_partial += f["n_partial"]
        w = max(est, E)
        out.append({"qId": qid, "src_est": est, "src_log_est": np.log1p(est),
                    "src_bound_rel": width / max(est, 1.0), "src_edge_share": (est - lo) / w,
                    "src_edge_contrast": contrast / w, "src_in_out": in_out / w, "src_empty_partial": empty / w,
                    "src_n_partial": n_partial})
    return pd.DataFrame(out)


def build_table_v1(run_dir, dataset_dir, **kw):
    """features.build_table plus the source-level and merge-shift inputs."""
    df = build_table(run_dir, dataset_dir, **kw)
    dropped = df.attrs.get("dropped_small_exact", 0)
    df = df.merge(source_features(run_dir, dataset_dir), on="qId", how="left")
    df["merge_shift"] = (df.est - df.srcSumEst) / np.maximum(df.srcSumEst, 1.0)
    df["abs_merge_shift"] = df.merge_shift.abs()
    df.attrs["dropped_small_exact"] = dropped
    return df


def check_source_estimates(df, tol=1e-3):
    """Largest relative gap between the Python no-merge estimate and Java's srcSumEst; must be below tol."""
    gap = float(((df.src_est - df.srcSumEst).abs() / np.maximum(df.srcSumEst, 1.0)).max())
    if gap > tol:
        raise AssertionError(f"source features disagree with Java's srcSumEst by {gap}")
    return gap
