"""Gate artifacts for the Java query path: per-merge-rate cutoffs and model.properties.

Java's predictor.GatedRangeQuery answers a query from the merged synopsis when the model's estimate is at or
below a cutoff, and reads raw data otherwise. The cutoff is not in model.txt; it is calibrated here, on the
calibration split (the same 50/20/30 query split as notebook 02 / gate_eval.split_ids), one cutoff per merge
rate, for each error limit eps at violation rate delta (gate_eval.cutoff).

Writes to out_dir:
  cutoffs.csv      mergeRate,eps,delta,cutoff            (java: -cutoffs, with -eps / -delta)
  model.properties label.transform and gate.cutoff       (java reads it next to model.txt)
  gate_report.csv  held-out check: per merge rate and eps, coverage and violation on the test queries

Use from a notebook after training:

    from gate_artifacts import write_gate_artifacts
    write_gate_artifacts(model, FEATURES, df, tmp, seed)       # model: LGBMRegressor or Booster; df: build_table

or from the command line for any saved model (features picked from the model's own feature names):

    python ml/gate_artifacts.py out/ml/authors_range data/ml/authors_range model.txt out/gate
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_eval import DELTA, EPS, cutoff, gate, split_ids  # noqa: E402


def _predict(model, df, names, identity):
    booster = getattr(model, "booster_", model)
    pred = booster.predict(df[names])
    return pred if identity else np.exp(pred)


def write_gate_artifacts(model, names, df, out_dir, seed=0, eps_list=EPS, delta=DELTA, identity=False,
                         primary_eps=None):
    """Calibrate on the calibration queries, check on the test queries, write the three files. Returns the
    held-out report. `df` needs qId, mergeRate and relErr, plus the model's features."""
    os.makedirs(out_dir, exist_ok=True)
    _, ca_ids, te_ids = split_ids(df.qId, seed)
    ca, te = df[df.qId.isin(ca_ids)], df[df.qId.isin(te_ids)]
    s_ca, s_te = _predict(model, ca, names, identity), _predict(model, te, names, identity)
    err_ca, err_te = ca.relErr.to_numpy(), te.relErr.to_numpy()
    rate_ca, rate_te = ca.mergeRate.to_numpy(), te.mergeRate.to_numpy()

    cut_rows, report = [], []
    for m in np.sort(np.unique(rate_ca)):
        c, t = rate_ca == m, rate_te == m
        for eps in eps_list:
            t_cut = cutoff(s_ca[c], err_ca[c], eps, delta)
            cut_rows.append({"mergeRate": m, "eps": eps, "delta": delta, "cutoff": t_cut})
            cov, viol = gate(s_te[t], err_te[t], eps, t_cut) if t.any() else (0.0, 0.0)
            report.append({"mergeRate": m, "eps": eps, "delta": delta, "cutoff": t_cut, "test_queries": int(t.sum()),
                           "coverage": cov, "violation": viol})
    pd.DataFrame(cut_rows).to_csv(os.path.join(out_dir, "cutoffs.csv"), index=False)
    report = pd.DataFrame(report)
    report.to_csv(os.path.join(out_dir, "gate_report.csv"), index=False)

    eps0 = eps_list[0] if primary_eps is None else primary_eps
    pooled = float(cutoff(s_ca, err_ca, eps0, delta))
    with open(os.path.join(out_dir, "model.properties"), "w") as f:
        f.write(f"label.transform={'identity' if identity else 'log'}\n")
        f.write(f"gate.cutoff={pooled!r}\n")          # pooled over merge rates (may be -inf): fallback for rates without a row
        f.write(f"gate.eps={eps0}\ngate.delta={delta}\n")
    return report


def _features_for(names, run_dir, dataset_dir):
    from features import build_table
    from features_v1 import SHIFT_FEATURES, SOURCE_FEATURES, build_table_v1
    needs_v1 = any(n in SOURCE_FEATURES + SHIFT_FEATURES for n in names)
    return (build_table_v1 if needs_v1 else build_table)(run_dir, dataset_dir)


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("dataset_dir")
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0, help="split seed; use the one the model was trained with")
    ap.add_argument("--eps", type=float, nargs="+", default=EPS)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--identity", action="store_true", help="model predicts the error itself, not its log")
    a = ap.parse_args()
    booster = lgb.Booster(model_file=a.model)
    names = booster.feature_name()
    df = _features_for(names, a.run_dir, a.dataset_dir)
    report = write_gate_artifacts(booster, names, df, a.out_dir, a.seed, a.eps, a.delta, a.identity)
    pooled = report.groupby("eps").apply(
        lambda g: pd.Series({"coverage": np.average(g.coverage, weights=g.test_queries),
                             "violation": np.average(g.violation, weights=g.test_queries)}), include_groups=False)
    print(f"wrote cutoffs.csv, model.properties, gate_report.csv to {a.out_dir}")
    print("held-out, weighted over merge rates:\n" + pooled.round(4).to_string())


if __name__ == "__main__":
    main()
