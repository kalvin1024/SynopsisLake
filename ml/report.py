"""Markdown data report for one ErrorPredictionLogExp run (formats: ml/SCHEMA.md).

    from report import build_report
    md = build_report(run_dir, data_dir, authors_avg="range_query/dataDrivenQuery/results/DataDrivenQueryAvgRelError.txt")

    python ml/report.py out/ml/authors_range data/ml/authors_range [--authors-avg FILE] [--out report.md]

Section 2 (error by merge rate) uses every ARM row, like the authors' averages. Sections 3 and 4 use
the training table from features.build_table (queries with exact count below MIN_EXACT dropped).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import MIN_EXACT, build_table, check_estimates  # noqa: E402

EPS = [0.10, 0.25]


def md_table(df, digits=3, col_digits=None):
    """Markdown table; integer columns stay integers, floats get `digits` decimals (per column: col_digits)."""
    cols = list(df.columns)
    col_digits = col_digits or {}

    def fmt(v, d):
        if isinstance(v, (bool, np.bool_)):
            return str(v)
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return f"{v:.{d}f}" if np.isfinite(v) else str(v)
        return str(v).replace("|", "\\|")

    numeric = [pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) for c in cols]
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v, col_digits.get(c, digits)) for c, v in zip(cols, row)) + " |")
    return "\n".join(lines)


ERROR_AGG = dict(mean_relErr=("relErr", "mean"), median_relErr=("relErr", "median"),
                 share_le_10pct=("relErr", lambda e: (e <= 0.10).mean()),
                 share_le_25pct=("relErr", lambda e: (e <= 0.25).mean()))


def load_authors_avg(path):
    """DataDrivenQueryAvgRelError.txt -> MergeRate(%), ourLH."""
    a = pd.read_csv(path)
    return a[["MergeRate(%)", "ourLH"]]


def _input_checks(rows, queries, df, gap):
    n_q, n_rates = len(queries), rows.mergeRate.nunique()
    gt = queries[4]
    label = df.label.to_numpy()
    checks = pd.DataFrame([
        ("queries", n_q),
        ("exact count min", int(gt.min())),
        ("exact count median", float(gt.median())),
        ("exact count mean", float(gt.mean())),
        ("exact count max", int(gt.max())),
        (f"queries with exact count < {MIN_EXACT}", int((gt < MIN_EXACT).sum())),
        ("merge rates", n_rates),
        ("ARM rows", len(rows)),
        ("queries x merge rates", n_q * n_rates),
        ("rows match queries x merge rates", len(rows) == n_q * n_rates),
        (f"rows after min-count rule (exact >= {MIN_EXACT})", len(df)),
        ("check_estimates max abs(est_py - est)", f"{gap:.2e}"),
        ("non-finite labels", int((~np.isfinite(label)).sum())),
    ], columns=["check", "value"])
    checks["value"] = checks["value"].astype(object)
    return checks


def _by_merge_rate(rows, authors_avg):
    t = (rows.groupby("mergeRate")
         .agg(lonRes=("lonRes", "first"), latRes=("latRes", "first"), alpha=("betaAlpha", "first"), **ERROR_AGG)
         .reset_index().sort_values("mergeRate", ascending=False))
    t.insert(0, "mergeRate_pct", (t.pop("mergeRate") * 100).round(1))
    note = ""
    if authors_avg is not None and os.path.exists(authors_avg):
        a = load_authors_avg(authors_avg)
        t = t.merge(a.rename(columns={"MergeRate(%)": "mergeRate_pct", "ourLH": "authors_ourLH"}),
                    on="mergeRate_pct", how="left")
        t["diff"] = (t.mean_relErr - t.authors_ourLH).abs()
        missing = int(t.authors_ourLH.isna().sum())
        note = (f"\nMax |mean_relErr - authors ourLH|: **{t['diff'].max():.3f}**"
                + (f" ({missing} merge rates missing from the authors' file)" if missing else "") + "\n")
    elif authors_avg is not None:
        note = f"\nAuthors' file not found: `{authors_avg}`\n"
    return t, note


def _by_scale(df):
    return (df.groupby("scale")
            .agg(queries=("qId", "nunique"), rows=("qId", "size"), side_frac=("side_frac", "first"), **ERROR_AGG)
            .reset_index())


def _baselines(df):
    out = []
    for name, score in [("skew_pred", df.skew_pred), ("bound_rel", df.bound_rel), ("-area_cells", -df.area_cells)]:
        r = {"score": name, "spearman": spearmanr(score, df.relErr).correlation}
        for eps in EPS:
            bad = df.relErr > eps
            r[f"auc_err_gt_{int(eps * 100)}pct"] = roc_auc_score(bad, score) if 0 < bad.mean() < 1 else float("nan")
        out.append(r)
    return pd.DataFrame(out)


def build_report(run_dir, data_dir, authors_avg=None, v0_csv=None, importance_csv=None):
    """Markdown report for run_dir (rows.csv, arm_*.txt, src_*.txt) and data_dir (queries-*)."""
    rows = pd.read_csv(os.path.join(run_dir, "rows.csv"))
    rows = rows[rows.method == "ARM"]
    queries = pd.read_csv(os.path.join(data_dir, "queries-Ans.txt"), header=None)
    df = build_table(run_dir, data_dir)
    gap = float((df.est_py - df.est).abs().max())
    try:
        check_estimates(df)
        gap_ok = "passes"
    except AssertionError:
        gap_ok = "**FAILS**"

    v0_csv = v0_csv or os.path.join(run_dir, "report", "v0_results.csv")
    importance_csv = importance_csv or os.path.join(run_dir, "report", "feature_importance.csv")

    md = [f"# Data report: `{os.path.basename(os.path.normpath(run_dir))}`", "",
          f"Run folder `{run_dir}`, dataset folder `{data_dir}`.", "",
          "## 1. Input checks", "", md_table(_input_checks(rows, queries, df, gap)), "",
          f"check_estimates {gap_ok} (tolerance 1e-3).", ""]

    t, note = _by_merge_rate(rows, authors_avg)
    md += ["## 2. Error by merge rate", "",
           "All ARM rows (no min-count rule), so `mean_relErr` is comparable to the authors' `ourLH`.", "",
           md_table(t, col_digits={"mergeRate_pct": 1}), note]

    md += ["## 3. Error by query scale", "", f"Training table (exact count >= {MIN_EXACT}), all merge rates.", "",
           md_table(_by_scale(df)), ""]

    alpha = rows.groupby("mergeRate").betaAlpha.first()
    md += ["## 4. Do simple scores track the error?", "",
           "Spearman rank correlation with relErr, and AUC for flagging rows with error above 10% / 25% "
           "(higher score = predicted worse).", "",
           md_table(_baselines(df)), "",
           f"Beta alpha across merge rates: min {alpha.min():.3f}, max {alpha.max():.3f} "
           f"({alpha.nunique()} distinct values for {len(alpha)} merge rates).", ""]

    md += ["## 5. Model v0", ""]
    if os.path.exists(v0_csv):
        md += ["Gate check on test queries (cutoff from calibration queries, at most 5% violations).", "",
               md_table(pd.read_csv(v0_csv), col_digits={"eps": 2}), ""]
    else:
        md += [f"No v0 results (`{v0_csv}` not found).", ""]
    if os.path.exists(importance_csv):
        imp = pd.read_csv(importance_csv).sort_values("gain", ascending=False).head(10)
        md += ["Top features by gain:", "", md_table(imp, digits=1), ""]
    return "\n".join(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("data_dir")
    ap.add_argument("--authors-avg")
    ap.add_argument("--v0-csv")
    ap.add_argument("--importance-csv")
    ap.add_argument("--out")
    a = ap.parse_args()
    text = build_report(a.run_dir, a.data_dir, a.authors_avg, a.v0_csv, a.importance_csv)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        print(text)
