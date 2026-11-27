"""Publication tables and a short interpretation for the matched-budget run."""

from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "source_shape": "H + Source Shape",
    "fine_reference": "H + fine-grid reference",
    "fine_direct": "Fine grid, direct answer",
    "exact": "Always exact",
}
DATASET_LABELS = {"authors_range": "Authors’ Range", "europe_range": "Europe"}
RISK_LABELS = {"met": "Met", "missed": "Missed", "undefined": "Undefined", "not_applicable": "—"}
PAPER_COLUMNS = [
    "dataset", "seed", "method", "merge_rate", "queries", "total_budget_bytes", "used_bytes",
    "accepted_count", "coverage", "accepted_bad_count", "violation", "fallback_count",
    "batch_median_s", "exact_batch_median_s", "seconds_saved_vs_exact",
]


def _fixed_h_summary(rows, delta_pct):
    ss = rows.loc[rows.method.eq("source_shape")].set_index("seed", verify_integrity=True)
    ref = rows.loc[rows.method.eq("fine_reference")].set_index("seed", verify_integrity=True)
    if ss.empty or set(ss.index) != set(ref.index):
        raise ValueError("The fixed-H comparison requires the same seeds for Source Shape and fine reference")
    paired = ss[["risk_target_status", "accepted_count"]].join(
        ref[["risk_target_status", "accepted_count"]], lsuffix="_ss", rsuffix="_ref")
    ss_met = paired.risk_target_status_ss.eq("met")
    ref_met = paired.risk_target_status_ref.eq("met")
    both_met = ss_met & ref_met
    result = (
        f"With H fixed, Source Shape met the {delta_pct}% violation target in "
        f"{ss_met.sum()}/{len(paired)} seeds and fine-grid reference in "
        f"{ref_met.sum()}/{len(paired)}. "
    )
    if both_met.any():
        counts = paired.loc[both_met]
        diff = counts.accepted_count_ss - counts.accepted_count_ref
        if diff.gt(0).all():
            result += f"Source Shape accepted more queries in all {len(diff)} seeds where both met the target."
        elif diff.lt(0).all():
            result += f"Fine-grid reference accepted more queries in all {len(diff)} seeds where both met the target."
        elif diff.eq(0).all():
            result += f"Acceptance counts were equal in all {len(diff)} seeds where both met the target."
        else:
            result += (
                f"Among the {len(diff)} seeds where both met the target, Source Shape accepted more in "
                f"{diff.gt(0).sum()}, fine-grid reference accepted more in {diff.lt(0).sum()}, "
                f"and {diff.eq(0).sum()} were tied."
            )
    else:
        result += "No seed had both methods meet the target."
    return result


def _comparison_paragraph(table, delta_pct):
    results = []
    for dataset, rows in table.groupby("dataset", sort=False):
        direct = rows.loc[rows.method.eq("fine_direct")]
        exact = rows.loc[rows.method.eq("exact")].iloc[0]
        if direct.empty:
            raise ValueError("The matched-budget report requires a fine-grid direct answer")
        direct_met = direct.risk_target_status.eq("met").sum()
        saved = direct.seconds_saved_vs_exact.mean()
        timing = f"saved {saved:.2f} s per batch" if saved >= 0 else f"took {-saved:.2f} s longer per batch"
        risk = direct.violation.dropna()
        if risk.empty:
            risk_text = "violation rate was undefined"
        else:
            risk_text = f"violation averaged {risk.mean():.2%} across seeds with accepted answers"
        results.append(
            f"**{DATASET_LABELS.get(dataset, dataset)}** ({exact.total_budget_bytes:,} byte cap). "
            + _fixed_h_summary(rows, delta_pct)
            + f" Fine-grid direct answers covered {direct.coverage.mean():.2%} of queries on average. "
            f"The {risk_text}. The method met the target in {direct_met}/{len(direct)} seeds "
            f"and averaged {direct.batch_median_s.mean():.2f} s per batch. Compared with the "
            f"{exact.batch_median_s:.2f} s always-exact baseline, it {timing}."
        )
    return "\n\n".join(results) + "\n\nAll seeds reuse the same eastern test queries. Target status uses each seed's observed violation rate."


def write_report(root, summary, delta):
    """Write a row per dataset, seed and method, with one always-exact row per dataset."""
    root = Path(root)
    table = summary[PAPER_COLUMNS].copy()
    table["seed"] = table.seed.astype("Int64")
    table["total_budget_bytes"] = table.total_budget_bytes.astype("int64")
    table["used_bytes"] = table.used_bytes.astype("int64")
    if table.duplicated(["dataset", "seed", "method"]).any():
        raise ValueError("Duplicate dataset/seed/method in the matched-budget summary")
    exact = table.method.eq("exact")
    if exact.sum() != table.dataset.nunique() or table.loc[exact, "dataset"].nunique() != table.dataset.nunique():
        raise ValueError("The matched-budget report requires one always-exact baseline per dataset")

    table["delta_target"] = delta
    table["risk_target_status"] = "undefined"
    measured_risk = ~exact & table.accepted_count.gt(0) & table.violation.notna()
    table.loc[measured_risk & table.violation.le(delta), "risk_target_status"] = "met"
    table.loc[measured_risk & table.violation.gt(delta), "risk_target_status"] = "missed"
    table.loc[exact, "risk_target_status"] = "not_applicable"
    table.to_csv(root / "paper_table.csv", index=False)

    delta_pct = f"{delta:.0%}".rstrip("%")
    header = [
        "Dataset", "Seed", "Method", "Storage cap (bytes)", "Used (bytes)",
        "Accepted", "Coverage", "Accepted violations", "Violation rate", f"{delta_pct}% target",
        "Exact fallbacks", "Batch (s)", "Always exact (s)", "Saved vs. exact (s)",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in table.itertuples(index=False):
        cells = [
            DATASET_LABELS.get(row.dataset, row.dataset),
            "—" if pd.isna(row.seed) else str(row.seed), METHOD_LABELS[row.method],
            f"{row.total_budget_bytes:,}", f"{row.used_bytes:,}",
            f"{row.accepted_count:,}", f"{row.coverage:.2%}", f"{row.accepted_bad_count:,}",
            "—" if pd.isna(row.violation) else f"{row.violation:.2%}", RISK_LABELS[row.risk_target_status],
            f"{row.fallback_count:,}", f"{row.batch_median_s:.2f}", f"{row.exact_batch_median_s:.2f}",
            f"{row.seconds_saved_vs_exact:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    (root / "paper_table.md").write_text(
        "# Held-out region results at the Source Shape storage budgets\n\n"
        "Both H methods return the original merged estimate when accepted. The direct method returns its "
        "fine-grid estimate. Scores use frozen west calibration and fresh east queries. Violation rate is "
        "the fraction of accepted estimates whose relative error exceeds 25%. Zero accepts leave that rate "
        "undefined. Always exact scans every query and has no approximate accepts. Batch time is the median "
        "of the Java timing repeats, including artifact loading, scoring, actual fallback and output conversion. "
        "The target status uses the unrounded observed rate.\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8",
    )
    (root / "matched_analysis_en.md").write_text(_comparison_paragraph(table, delta_pct) + "\n", encoding="utf-8")
