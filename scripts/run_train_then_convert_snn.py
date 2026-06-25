#!/usr/bin/env python3
"""Run the core train-then-convert YOLOv26n-SNN protocol.

This is the thesis-facing runner: it evaluates an already trained YOLOv26n ANN
checkpoint, converts the selected feature blocks into calibrated spiking nodes,
and writes both raw aggregate metrics and a clean ANN-vs-SNN comparison table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paths import RESULTS_DIR
from run_hybrid_batch import (
    MODALITIES,
    default_checkpoint,
    default_data_yaml,
    format_template,
    job_label,
)
from run_hybrid_snn import (
    collect_activation_stats,
    evaluate,
    parse_csv_values,
    write_activation_stats,
)
from summarize_ann_snn_comparison import comparison_row


DEFAULT_SCOPE = "range5-10+range13-13+range17-22"


def resolve_data_yaml(args, modality: str, fold: int | None) -> Path:
    if args.data_template:
        return format_template(args.data_template, args.dataset, modality, args.activation, args.tag, fold)
    return default_data_yaml(args.dataset, modality, fold)


def resolve_checkpoint(args, modality: str, fold: int | None) -> Path:
    if args.checkpoint_template:
        return format_template(args.checkpoint_template, args.dataset, modality, args.activation, args.tag, fold)
    return default_checkpoint(args.dataset, modality, args.activation, args.tag, fold)


def default_out_dir(dataset: str) -> Path:
    group = "single_chunk" if dataset == "single" else "kfold"
    return RESULTS_DIR / group / "train_then_convert_snn"


def write_rows(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_comparison(rows: list[dict]) -> None:
    for row in rows:
        fold = f" fold={row['fold']}" if row["fold"] != "" else ""
        print(
            f"{row['modality']} {row['split']}{fold}: "
            f"ANN={row['ann_map50_95']:.4f} SNN={row['snn_map50_95']:.4f} "
            f"retention={row['map50_95_retention']:.2%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Core train-then-convert YOLOv26n-SNN protocol")
    parser.add_argument("--dataset", choices=["single", "kfold"], default="single")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--mod", choices=["LF", "HF", "both"], default="both")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--activation", choices=["silu", "relu"], default="relu")
    parser.add_argument("--tag", default="chunk2")
    parser.add_argument("--checkpoint-template", default=None)
    parser.add_argument("--data-template", default=None)
    parser.add_argument("--skip-missing", action="store_true")

    parser.add_argument("--snn-scope", default=DEFAULT_SCOPE)
    parser.add_argument("--snn-threshold", type=float, default=0.4)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--source-activation", choices=["SiLU", "ReLU", "both"], default="ReLU")
    parser.add_argument("--spike-scale", choices=["one", "threshold"], default="threshold")
    parser.add_argument("--calibration-stat", choices=["none", "p95", "p99", "p999", "max"], default="p99")
    parser.add_argument("--calibration-granularity", choices=["global", "channel"], default="channel")
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--calibration-samples", type=int, default=20000)

    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--aggregate-out", type=Path, default=None)
    parser.add_argument("--comparison-out", type=Path, default=None)
    parser.add_argument("--label", default="YOLOv26n-SNN train-then-convert channel-wise")
    args = parser.parse_args()

    modalities = MODALITIES if args.mod == "both" else [args.mod]
    folds = [None] if args.dataset == "single" else args.folds
    splits = parse_csv_values(args.splits, str)
    out_dir = args.out_dir or default_out_dir(args.dataset)

    aggregate_rows: list[dict] = []
    comparison_rows: list[dict] = []

    for fold in folds:
        for modality in modalities:
            data_yaml = resolve_data_yaml(args, modality, fold)
            checkpoint = resolve_checkpoint(args, modality, fold)
            if not data_yaml.exists() or not checkpoint.exists():
                message = f"Missing data/checkpoint: data={data_yaml} checkpoint={checkpoint}"
                if args.skip_missing:
                    print(f"[skip] {message}")
                    continue
                raise FileNotFoundError(message)

            calibration_stats = None
            if args.calibration_stat != "none":
                stats_out = out_dir / f"{job_label(args.dataset, modality, args.calibration_split, args.tag, fold)}_activation_stats.csv"
                print(f"\nStats: {checkpoint} | {data_yaml} | split={args.calibration_split}")
                calibration_stats = collect_activation_stats(
                    checkpoint=checkpoint,
                    data_yaml=data_yaml,
                    source_activation=args.source_activation,
                    split=args.calibration_split,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    sample_limit=args.calibration_samples,
                    granularity=args.calibration_granularity,
                )
                write_activation_stats(calibration_stats, stats_out)
                print(f"  saved stats: {stats_out}")

            for split in splits:
                print(f"\nANN eval: dataset={args.dataset} fold={fold} mod={modality} split={split}")
                ann_row = evaluate(
                    checkpoint=checkpoint,
                    data_yaml=data_yaml,
                    scope="none",
                    threshold=1.0,
                    source_activation=args.source_activation,
                    timesteps=1,
                    split=split,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    spike_scale=args.spike_scale,
                    calibration_stat=args.calibration_stat,
                    calibration_stats=calibration_stats,
                    calibration_granularity=args.calibration_granularity,
                )
                ann_row = {
                    "dataset": args.dataset,
                    "fold": "" if fold is None else fold,
                    "modality": modality,
                    "tag": args.tag if args.dataset == "single" else "",
                    "data_yaml": str(data_yaml),
                    "checkpoint": str(checkpoint),
                    "model_variant": "ANN",
                    **ann_row,
                }
                aggregate_rows.append(ann_row)
                print(f"  ANN mAP50={ann_row['map50']:.4f} mAP50-95={ann_row['map50_95']:.4f}")

                print(
                    f"SNN eval: dataset={args.dataset} fold={fold} mod={modality} split={split} "
                    f"scope={args.snn_scope} threshold={args.snn_threshold} T={args.timesteps}"
                )
                snn_row = evaluate(
                    checkpoint=checkpoint,
                    data_yaml=data_yaml,
                    scope=args.snn_scope,
                    threshold=args.snn_threshold,
                    source_activation=args.source_activation,
                    timesteps=args.timesteps,
                    split=split,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    spike_scale=args.spike_scale,
                    calibration_stat=args.calibration_stat,
                    calibration_stats=calibration_stats,
                    calibration_granularity=args.calibration_granularity,
                )
                snn_row = {
                    "dataset": args.dataset,
                    "fold": "" if fold is None else fold,
                    "modality": modality,
                    "tag": args.tag if args.dataset == "single" else "",
                    "data_yaml": str(data_yaml),
                    "checkpoint": str(checkpoint),
                    "model_variant": "SNN",
                    **snn_row,
                }
                aggregate_rows.append(snn_row)
                comparison_rows.append(comparison_row(ann_row, snn_row, args.label))
                print(f"  SNN mAP50={snn_row['map50']:.4f} mAP50-95={snn_row['map50_95']:.4f}")

    if not aggregate_rows:
        print("No results to write.")
        return

    aggregate_out = args.aggregate_out or out_dir / f"{args.dataset}_train_then_convert_snn_aggregate.csv"
    comparison_out = args.comparison_out or out_dir / f"{args.dataset}_ann_vs_snn_comparison.csv"
    write_rows(aggregate_rows, aggregate_out)
    write_rows(comparison_rows, comparison_out)
    print(f"\nSaved aggregate: {aggregate_out}")
    print(f"Saved comparison: {comparison_out}")
    print_comparison(comparison_rows)


if __name__ == "__main__":
    main()
