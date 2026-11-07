"""Check that the Java learned error predictor reproduces LightGBM's own predictions.

Run ErrorPredictionLogExp with -modelPath <model.txt> (adds a learnedErr column to rows.csv), then:

    python ml/check_java_parity.py out/ml/authors_range data/ml/authors_range model.txt

The Python side builds the features from the exported histograms (features.py / features_v1.py) and calls
Booster.predict; the Java side computed them itself. Both estimates are exp(prediction) (log label).

Java and numpy sum in a different order, so features differ in the last bit (~1e-16). A model that has split
on that noise (sliding-window queries give mathematically equal features that differ only by rounding) then
sends a few rows down a different path. So the check is on the fraction of rows beyond --tol, and on what the
gate cares about: rank agreement and AUC against the true error. It fails when more than --max-bad-frac of
the rows differ or the Spearman correlation is below --min-spearman.
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_table  # noqa: E402
from features_v1 import SHIFT_FEATURES, SOURCE_FEATURES, build_table_v1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("dataset_dir")
    ap.add_argument("model")
    ap.add_argument("--tol", type=float, default=1e-6, help="relative difference that counts as a mismatch")
    ap.add_argument("--max-bad-frac", type=float, default=0.005)
    ap.add_argument("--min-spearman", type=float, default=0.9999)
    ap.add_argument("--identity", action="store_true", help="model predicts the error itself, not its log")
    a = ap.parse_args()

    booster = lgb.Booster(model_file=a.model)
    names = booster.feature_name()
    needs_v1 = any(n in SOURCE_FEATURES + SHIFT_FEATURES for n in names)
    df = (build_table_v1 if needs_v1 else build_table)(a.run_dir, a.dataset_dir)
    pred = booster.predict(df[names])
    df["py_err"] = pred if a.identity else np.exp(pred)

    rows = pd.read_csv(os.path.join(a.run_dir, "rows.csv"), usecols=["lonRes", "qId", "learnedErr"])
    m = df[["lonRes", "qId", "relErr", "py_err"]].merge(rows, on=["lonRes", "qId"], how="left")
    if m.learnedErr.isna().any():
        sys.exit("rows.csv has no learnedErr for some rows: run ErrorPredictionLogExp with -modelPath")
    rel = (m.learnedErr - m.py_err).abs() / np.maximum(m.py_err.abs(), 1e-12)
    bad = float((rel > a.tol).mean())
    rho = m.learnedErr.corr(m.py_err, method="spearman")
    print(f"{len(m)} rows, {len(names)} features, v1 features: {needs_v1}")
    print(f"relative difference: median {rel.median():.3e}, p99 {rel.quantile(.99):.3e}, max {rel.max():.3e}")
    print(f"rows beyond tol {a.tol:g}: {int((rel > a.tol).sum())} ({bad:.3%}); spearman {rho:.6f}")
    for eps in (0.10, 0.25):
        y = m.relErr > eps
        print(f"AUC for relErr > {eps}: python {roc_auc_score(y, m.py_err):.5f}, java {roc_auc_score(y, m.learnedErr):.5f}")
    if bad > a.max_bad_frac or rho < a.min_spearman:
        sys.exit("FAIL: Java predictions do not match LightGBM")
    print("OK")


if __name__ == "__main__":
    main()
