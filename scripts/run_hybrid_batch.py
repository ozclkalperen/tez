#!/usr/bin/env python3
"""
run_hybrid_batch.py — Hybrid SNN ayarını single-chunk veya k-fold işlerine uygular.

Bu script, pilotta bulunan aynı SNN ayarını daha sonra 5-fold protokole
taşımak için ince bir sarmalayıcıdır. Tek tek `run_hybrid_snn.py` komutu
yazmak yerine data/checkpoint yollarını deney tasarımına göre üretir ve tek
bir aggregate CSV kaydeder.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paths import DATASETS_DIR, RESULTS_DIR, RUNS_DIR
from run_hybrid_snn import (
    collect_activation_stats,
    evaluate,
    parse_csv_values,
    write_activation_stats,
)


SAFE_SCOPE = "range5-10+range13-13+range17-22"
MODALITIES = ["LF", "HF"]


def activation_suffix(activation: str) -> str:
    return "" if activation == "silu" else f"_{activation}"


def default_data_yaml(dataset: str, modality: str, fold: int | None) -> Path:
    if dataset == "single":
        return DATASETS_DIR / "subpipe_single_chunk" / modality / "data.yaml"
    if fold is None:
        raise ValueError("kfold dataset için fold gerekli")
    return DATASETS_DIR / "subpipe_kfold" / f"fold_{fold}" / modality / "data.yaml"


def default_checkpoint(dataset: str, modality: str, activation: str, tag: str, fold: int | None) -> Path:
    suffix = activation_suffix(activation)
    if dataset == "single":
        return RUNS_DIR / "single_chunk" / f"{tag}_{modality}{suffix}" / "weights" / "best.pt"
    if fold is None:
        raise ValueError("kfold dataset için fold gerekli")
    return RUNS_DIR / "kfold" / f"fold{fold}_{modality}{suffix}" / "weights" / "best.pt"


def format_template(template: str, dataset: str, modality: str, activation: str, tag: str, fold: int | None) -> Path:
    value = template.format(
        dataset=dataset,
        mod=modality,
        modality=modality,
        activation=activation,
        suffix=activation_suffix(activation),
        tag=tag,
        fold="" if fold is None else fold,
    )
    return Path(value)


def resolve_data_yaml(args, modality: str, fold: int | None) -> Path:
    if args.data_template:
        return format_template(args.data_template, args.dataset, modality, args.activation, args.tag, fold)
    return default_data_yaml(args.dataset, modality, fold)


def resolve_checkpoint(args, modality: str, fold: int | None) -> Path:
    if args.checkpoint_template:
        return format_template(args.checkpoint_template, args.dataset, modality, args.activation, args.tag, fold)
    return default_checkpoint(args.dataset, modality, args.activation, args.tag, fold)


def output_dir(args) -> Path:
    if args.out_dir:
        return args.out_dir
    group = "single_chunk" if args.dataset == "single" else "kfold"
    return RESULTS_DIR / group / "hybrid_snn"


def job_label(dataset: str, modality: str, split: str, tag: str, fold: int | None) -> str:
    if dataset == "single":
        return f"{tag}_{modality}_{split}"
    return f"fold{fold}_{modality}_{split}"


def write_rows(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch hybrid SNN evaluation")
    parser.add_argument("--dataset", choices=["single", "kfold"], default="single")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--mod", choices=["LF", "HF", "both"], default="both")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--activation", choices=["silu", "relu"], default="relu")
    parser.add_argument("--tag", default="chunk2", help="single dataset run adı öneki")
    parser.add_argument("--checkpoint-template", default=None)
    parser.add_argument("--data-template", default=None)
    parser.add_argument("--skip-missing", action="store_true")

    parser.add_argument("--scopes", default=f"none,{SAFE_SCOPE}")
    parser.add_argument("--thresholds", default="0.6")
    parser.add_argument("--source-activation", choices=["SiLU", "ReLU", "both"], default="ReLU")
    parser.add_argument("--spike-scale", choices=["one", "threshold"], default="threshold")
    parser.add_argument("--calibration-stat", choices=["none", "p95", "p99", "p999", "max"], default="p99")
    parser.add_argument("--calibration-granularity", choices=["global", "channel"], default="global")
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--calibration-samples", type=int, default=20000)
    parser.add_argument("--timesteps", default="16")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    modalities = MODALITIES if args.mod == "both" else [args.mod]
    folds = [None] if args.dataset == "single" else args.folds
    splits = parse_csv_values(args.splits, str)
    scopes = parse_csv_values(args.scopes, str)
    thresholds = parse_csv_values(args.thresholds, float)
    timesteps = parse_csv_values(args.timesteps, int)
    out_root = output_dir(args)
    aggregate_rows: list[dict] = []

    for fold in folds:
        for modality in modalities:
            data_yaml = resolve_data_yaml(args, modality, fold)
            checkpoint = resolve_checkpoint(args, modality, fold)
            if not data_yaml.exists() or not checkpoint.exists():
                message = f"Eksik job: data={data_yaml} checkpoint={checkpoint}"
                if args.skip_missing:
                    print(f"[skip] {message}")
                    continue
                raise FileNotFoundError(message)

            calibration_stats = None
            if args.calibration_stat != "none":
                stats_out = out_root / f"{job_label(args.dataset, modality, args.calibration_split, args.tag, fold)}_activation_stats.csv"
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
                for scope in scopes:
                    th_values = [1.0] if scope == "none" else thresholds
                    t_values = [1] if scope == "none" else timesteps
                    for threshold in th_values:
                        for timestep in t_values:
                            print(
                                f"\nEval: dataset={args.dataset} fold={fold} mod={modality} "
                                f"split={split} scope={scope} threshold={threshold} T={timestep}"
                            )
                            row = evaluate(
                                checkpoint=checkpoint,
                                data_yaml=data_yaml,
                                scope=scope,
                                threshold=threshold,
                                source_activation=args.source_activation,
                                timesteps=timestep,
                                split=split,
                                imgsz=args.imgsz,
                                batch=args.batch,
                                device=args.device,
                                spike_scale=args.spike_scale,
                                calibration_stat=args.calibration_stat,
                                calibration_stats=calibration_stats,
                                calibration_granularity=args.calibration_granularity,
                            )
                            row = {
                                "dataset": args.dataset,
                                "fold": "" if fold is None else fold,
                                "modality": modality,
                                "tag": args.tag if args.dataset == "single" else "",
                                "data_yaml": str(data_yaml),
                                "checkpoint": str(checkpoint),
                                **row,
                            }
                            aggregate_rows.append(row)
                            print(f"  mAP50={row['map50']:.4f} mAP50-95={row['map50_95']:.4f}")

    if not aggregate_rows:
        print("Yazılacak sonuç yok.")
        return

    out = args.out or out_root / f"{args.dataset}_hybrid_snn_results.csv"
    write_rows(aggregate_rows, out)
    print(f"\nSaved aggregate: {out}")


if __name__ == "__main__":
    main()
