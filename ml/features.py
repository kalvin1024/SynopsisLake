"""Model v1 features and labels from one ErrorPredictionLogExp run (formats: ml/SCHEMA.md).

Every input is computed from the exported histograms and the query rectangle only,
never from raw data or the exact answer. The exact answer is used for the label.

    from features import build_table
    df = build_table(run_dir, dataset_dir)          # one row per (merge rate, query), ARM only
    X, y = df[FEATURES], df["label"]
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

E = 1e-9
MIN_EXACT = 10          # DEC2: queries with exact count below this are dropped
LABEL_FLOOR = 1e-3      # label = log(max(relErr, LABEL_FLOOR))

FEATURES = [
    # query shape
    "area_cells", "n_cells", "n_partial", "side_frac",
    # edge cells
    "log_est", "log_full", "bound_rel", "edge_share", "edge_contrast", "in_out", "empty_partial",
    # scale-free
    "est_frac", "density_ratio",
    # merge
    "merge_rate", "n_src_overlap", "src_res_min", "src_res_max",
    # SynopsisLake's own estimate
    "skew_pred", "log_alpha",
]


def load_hist(path):
    """Histogram export: line 1 merge rate, line 2 lon boundaries, line 3 lat boundaries,
    then one row of counts per lon bucket."""
    with open(path) as f:
        lines = f.read().splitlines()
    rate = float(lines[0])
    bx = np.array(lines[1].split(","), float)
    by = np.array(lines[2].split(","), float)
    counts = np.array([r.split(",") for r in lines[3:]], float)
    return rate, bx, by, counts


def _overlap(b, lo, hi):
    n = len(b) - 1
    i0 = int(np.clip(np.searchsorted(b, lo, "right") - 1, 0, n - 1))
    i1 = int(np.clip(np.searchsorted(b, hi, "left") - 1, 0, n - 1))
    left, right = b[i0:i1 + 1], b[i0 + 1:i1 + 2]
    frac = np.clip((np.minimum(hi, right) - np.maximum(lo, left)) / (right - left), 0, 1)
    return i0, i1, frac


def _query_features(q, bx, by, C, area, LC, med_density, total, sources):
    a, b, c, d = q
    i0, i1, fx = _overlap(bx, a, b)
    j0, j1, fy = _overlap(by, c, d)
    w = np.outer(fx, fy)
    sub = C[i0:i1 + 1, j0:j1 + 1]
    est = float((w * sub).sum())
    full = w >= 1 - 1e-12
    part = (w > 0) & ~full
    lo_count = float(sub[full].sum())
    hi_count = float(sub[w > 0].sum())
    pw = sub[part] + 1
    edge_contrast = float((LC[i0:i1 + 1, j0:j1 + 1][part] * pw).sum() / pw.sum()) if part.any() else 0.0
    I0, I1 = max(i0 - 1, 0), min(i1 + 2, C.shape[0])
    J0, J1 = max(j0 - 1, 0), min(j1 + 2, C.shape[1])
    a_in = area[i0:i1 + 1, j0:j1 + 1].sum()
    ring_c = C[I0:I1, J0:J1].sum() - sub.sum()
    ring_a = area[I0:I1, J0:J1].sum() - a_in
    d_in = sub.sum() / a_in
    d_out = ring_c / ring_a if ring_a > 0 else d_in
    over = [s for s in sources if s[0] < b and a < s[1] and s[2] < d and c < s[3]]
    return {
        "est_py": est,
        "area_cells": float(w.sum()),
        "n_cells": int((w > 0).sum()),
        "n_partial": int(part.sum()),
        "side_frac": (b - a) / (bx[-1] - bx[0]),
        "log_est": np.log1p(est),
        "log_full": np.log1p(lo_count),
        "bound_rel": (hi_count - lo_count) / max(est, 1.0),
        "edge_share": (est - lo_count) / max(est, E),
        "edge_contrast": edge_contrast,
        "in_out": float(np.log((d_in + E) / (d_out + E))),
        "empty_partial": float((sub[part] == 0).mean()) if part.any() else 0.0,
        "est_frac": est / max(total, 1.0),
        "density_ratio": float(np.log((d_in + E) / (med_density + E))),
        "n_src_overlap": len(over),
        "src_res_min": min((s[4] for s in over), default=0),
        "src_res_max": max((s[4] for s in over), default=0),
    }


def build_table(run_dir, dataset_dir, rows_csv=None, min_exact=MIN_EXACT):
    """Features + label for every ARM row of one run.

    run_dir      folder with arm_<res>.txt, src_<i>.txt (and rows.csv unless rows_csv is given)
    dataset_dir  folder with queries-Ans.txt and queries-meta.csv
    """
    rows = pd.read_csv(rows_csv or os.path.join(run_dir, "rows.csv"))
    rows = rows[rows.method == "ARM"].reset_index(drop=True)
    queries = pd.read_csv(os.path.join(dataset_dir, "queries-Ans.txt"), header=None).to_numpy()
    meta = pd.read_csv(os.path.join(dataset_dir, "queries-meta.csv"))

    sources = []
    for p in sorted(glob.glob(os.path.join(run_dir, "src_*.txt"))):
        _, sx, sy, _ = load_hist(p)
        sources.append((sx[0], sx[-1], sy[0], sy[-1], len(sx) - 1))

    out = []
    for res, grp in rows.groupby("lonRes"):
        rate, bx, by, C = load_hist(os.path.join(run_dir, f"arm_{res}.txt"))
        area = np.outer(np.diff(bx), np.diff(by))
        D = C / area
        LC = np.abs(np.log((D + E) / (uniform_filter(D, size=3, mode="nearest") + E)))
        nonempty = D[C > 0]
        med_density = float(np.median(nonempty)) if nonempty.size else 0.0
        total = float(C.sum())
        for r in grp.itertuples():
            f = _query_features(queries[r.qId, :4], bx, by, C, area, LC, med_density, total, sources)
            f["idx"] = r.Index
            out.append(f)
    feats = pd.DataFrame(out).set_index("idx")
    df = rows.join(feats)
    df = df.merge(meta[["qId", "scale"]], on="qId", how="left")
    df["merge_rate"] = df.mergeRate
    df["skew_pred"] = df.predErr
    df["log_alpha"] = np.log10(df.betaAlpha.clip(lower=1.0))
    df["label"] = np.log(np.maximum(df.relErr, LABEL_FLOOR))

    n_before = len(df)
    df = df[df["gt"] >= min_exact].reset_index(drop=True)   # df.gt is pandas' method, not the column
    df.attrs["dropped_small_exact"] = n_before - len(df)
    return df


def check_estimates(df, tol=1e-3):
    """Largest gap between the Python estimate and the Java estimate; must be below tol."""
    gap = float((df.est_py - df.est).abs().max())
    if gap > tol:
        raise AssertionError(f"feature code disagrees with Java estimate by {gap}")
    return gap


def dataset_name(path):
    return re.sub(r"[^A-Za-z0-9_-]", "_", os.path.basename(os.path.normpath(path)))
