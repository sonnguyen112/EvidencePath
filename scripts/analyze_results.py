#!/usr/bin/env python3
"""Aggregate per-question results and compute paired bootstrap comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

from evidencepath.datasets import read_records
from evidencepath.metrics import aggregate_prediction_rows, holm_adjust, paired_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-method", default="evidencepath")
    parser.add_argument("--baselines", nargs="+", default=["hipporag", "sentence"])
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.results.glob("*.jsonl")):
        rows.extend(read_records(path))
    if not rows:
        raise SystemExit(f"No JSONL result files found in {args.results}")

    groups: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), int(row["budget"]), str(row["ablation"]), str(row["method"]))].append(row)

    summaries = []
    for (dataset, budget, ablation, method), group in sorted(groups.items()):
        summaries.append(
            {
                "dataset": dataset,
                "budget": budget,
                "ablation": ablation,
                "method": method,
                **aggregate_prediction_rows(group),
            }
        )

    comparisons = []
    pending_p_values = []
    for (dataset, budget, ablation, method), reference_rows in sorted(groups.items()):
        if method != args.reference_method or ablation != "full":
            continue
        reference_by_id = {str(row["question_id"]): row for row in reference_rows}
        for baseline in args.baselines:
            baseline_rows = groups.get((dataset, budget, ablation, baseline), [])
            baseline_by_id = {str(row["question_id"]): row for row in baseline_rows}
            question_ids = sorted(set(reference_by_id) & set(baseline_by_id))
            if not question_ids:
                continue
            result = paired_bootstrap(
                [float(reference_by_id[item]["f1"]) for item in question_ids],
                [float(baseline_by_id[item]["f1"]) for item in question_ids],
                resamples=args.resamples,
                seed=args.seed,
            )
            comparison = {
                "dataset": dataset,
                "budget": budget,
                "ablation": ablation,
                "reference": args.reference_method,
                "baseline": baseline,
                **result,
            }
            comparisons.append(comparison)
            pending_p_values.append(float(result["p_value"]))

    adjusted = holm_adjust(pending_p_values)
    for comparison, adjusted_p in zip(comparisons, adjusted):
        comparison["holm_p_value"] = adjusted_p

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"summaries": summaries, "comparisons": comparisons}
    (args.output / "tables.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    with (args.output / "bootstrap.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in comparisons for key in row})
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(comparisons)


if __name__ == "__main__":
    main()

