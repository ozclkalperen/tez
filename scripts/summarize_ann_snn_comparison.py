#!/usr/bin/env python3
"""Create a clean ANN vs SNN comparison table from hybrid batch results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paths import RESULTS_DIR


DEFAULT_IN = (
    RESULTS_DIR
    / "single_chunk"
    / "hybrid_snn"
    / "batch_single_lf_hf_val_test_safe_channelwise_threshold_sweep_t16.csv"
)
DEFAULT_OUT = RESULTS_DIR / "single_chunk" / "yolov26n_ann_vs_snn_core_comparison.csv"
DEFAULT_SCOPE = "range5-10+range13-13+range17-22"


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def find_row(
    rows: list[dict[str, str]],
    modality: str,
    split: str,
    scope: str,
    threshold: float | None,
) -> dict[str, str]:
    matches = []
    for row in rows:
        if row["modality"] != modality or row["split"] != split or row["scope"] != scope:
            continue
        if threshold is not None and abs(float(row["threshold"]) - threshold) > 1e-9:
            continue
        matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for modality={modality}, split={split}, "
            f"scope={scope}, threshold={threshold}; found {len(matches)}"
        )
    return matches[0]


def comparison_row(ann: dict[str, str], snn: dict[str, str], label: str) -> dict[str, object]:
    ann_map = as_float(ann, "map50_95")
    snn_map = as_float(snn, "map50_95")
    ann_map50 = as_float(ann, "map50")
    snn_map50 = as_float(snn, "map50")
    retention = snn_map / ann_map if ann_map else 0.0
    return {
        "dataset": snn.get("dataset", ""),
        "fold": snn.get("fold", ""),
        "modality": snn["modality"],
        "split": snn["split"],
        "snn_label": label,
        "ann_scope": ann["scope"],
        "snn_scope": snn["scope"],
        "snn_threshold": snn["threshold"],
        "snn_timesteps": snn["timesteps"],
        "snn_replaced": snn["replaced"],
        "snn_calibration": f"{snn['calibration_granularity']}:{snn['calibration_stat']}",
        "ann_map50": round(ann_map50, 6),
        "snn_map50": round(snn_map50, 6),
        "delta_map50": round(snn_map50 - ann_map50, 6),
        "ann_map50_95": round(ann_map, 6),
        "snn_map50_95": round(snn_map, 6),
        "delta_map50_95": round(snn_map - ann_map, 6),
        "map50_95_retention": round(retention, 6),
    }


def write_rows(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ANN vs SNN comparison")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--snn-scope", default=DEFAULT_SCOPE)
    parser.add_argument("--snn-threshold", type=float, default=0.4)
    parser.add_argument("--label", default="YOLOv26n-SNN hybrid channel-wise")
    args = parser.parse_args()

    rows = load_rows(args.input)
    keys = sorted({(row["modality"], row["split"]) for row in rows})
    output = []
    for modality, split in keys:
        ann = find_row(rows, modality, split, "none", 1.0)
        snn = find_row(rows, modality, split, args.snn_scope, args.snn_threshold)
        output.append(comparison_row(ann, snn, args.label))

    write_rows(output, args.out)
    print(f"Saved: {args.out}")
    for row in output:
        print(
            f"{row['modality']} {row['split']}: "
            f"ANN={row['ann_map50_95']:.4f} SNN={row['snn_map50_95']:.4f} "
            f"retention={row['map50_95_retention']:.2%}"
        )


if __name__ == "__main__":
    main()
