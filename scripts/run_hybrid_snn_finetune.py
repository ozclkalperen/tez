#!/usr/bin/env python3
"""
run_hybrid_snn_finetune.py — Güvenli hybrid SNN blokları için küçük fine-tune.

Bu script, pilotta seçilen spiking blokları eğitim grafiğine dahil eder.
Amaç tam SNN eğitimi değil; analog bırakılan erken/head bölgeler sabit
kalırken spiking yapılan orta/geç blokların kısa fine-tune ile toparlanıp
toparlanmadığını ölçmektir.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
import torch

from paths import DATASETS_DIR, RESULTS_DIR, RUNS_DIR
from run_hybrid_snn import (
    HybridSNNModel,
    collect_activation_stats,
    replace_activations,
    write_activation_stats,
)


SAFE_SCOPE = "range5-10+range13-13+range17-22"
DEFAULT_CHECKPOINT = RUNS_DIR / "single_chunk" / "chunk2_LF_relu" / "weights" / "best.pt"
DEFAULT_DATA = DATASETS_DIR / "subpipe_single_chunk" / "LF" / "data.yaml"
DEFAULT_OUT = RESULTS_DIR / "single_chunk" / "hybrid_snn_finetune_lf.csv"
DEFAULT_STATS = RESULTS_DIR / "single_chunk" / "hybrid_snn_finetune_lf_activation_stats.csv"


def metrics_row(metrics, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_precision": round(float(metrics.box.mp), 6),
        f"{prefix}_recall": round(float(metrics.box.mr), 6),
        f"{prefix}_map50": round(float(metrics.box.map50), 6),
        f"{prefix}_map50_95": round(float(metrics.box.map), 6),
    }


def evaluate_yolo(yolo: YOLO, data: Path, split: str, imgsz: int, batch: int, device: str):
    return yolo.val(data=str(data), split=split, imgsz=imgsz, batch=batch, device=device, verbose=False, plots=False)


def write_result(row: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def augmentation_kwargs(mode: str) -> dict:
    if mode == "default":
        return {}
    if mode != "none":
        raise ValueError(f"Bilinmeyen augmentation modu: {mode}")
    return {
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "erasing": 0.0,
        "close_mosaic": 0,
    }


def materialize_trainable_tensors(model) -> None:
    """Convert inference-mode checkpoint tensors into regular tensors for training."""
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None:
                continue
            module._parameters[name] = torch.nn.Parameter(
                param.detach().clone(),
                requires_grad=True,
            )
        for name, buffer in list(module._buffers.items()):
            if buffer is None or not torch.is_tensor(buffer):
                continue
            module._buffers[name] = buffer.detach().clone()


def build_hybrid_model(args, train_timesteps: int) -> tuple[YOLO, int, list[float]]:
    stats = None
    if args.calibration_stat != "none":
        print(f"Collecting activation stats split={args.calibration_split} stat={args.calibration_stat}")
        stats = collect_activation_stats(
            checkpoint=args.checkpoint,
            data_yaml=args.data,
            source_activation=args.source_activation,
            split=args.calibration_split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            sample_limit=args.calibration_samples,
            granularity=args.calibration_granularity,
        )
        write_activation_stats(stats, args.stats_out)
        print(f"Saved stats: {args.stats_out}")

    yolo = YOLO(str(args.checkpoint))
    base = yolo.model
    materialize_trainable_tensors(base)
    replaced, thresholds = replace_activations(
        base=base,
        scope=args.scope,
        threshold=args.threshold,
        source_activation=args.source_activation,
        spike_scale=args.spike_scale,
        calibration_stat=args.calibration_stat,
        calibration_stats=stats,
        calibration_granularity=args.calibration_granularity,
        surrogate_alpha=args.surrogate_alpha,
        learn_thresholds=args.learn_thresholds,
    )
    yolo.model = HybridSNNModel(base, timesteps=train_timesteps)
    return yolo, replaced, thresholds


def trainer_for(model):
    class HybridSNNTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return model

    return HybridSNNTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid SNN fine-tune")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--scope", default=SAFE_SCOPE)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--source-activation", choices=["SiLU", "ReLU", "both"], default="ReLU")
    parser.add_argument("--spike-scale", choices=["one", "threshold"], default="threshold")
    parser.add_argument("--calibration-stat", choices=["none", "p95", "p99", "p999", "max"], default="p99")
    parser.add_argument("--calibration-granularity", choices=["global", "channel"], default="global")
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--calibration-samples", type=int, default=20000)
    parser.add_argument("--surrogate-alpha", type=float, default=4.0)
    parser.add_argument("--learn-thresholds", action="store_true")
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--train-timesteps", type=int, default=4)
    parser.add_argument("--eval-timesteps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--augmentation", choices=["default", "none"], default="default")
    parser.add_argument("--project", type=Path, default=RUNS_DIR / "single_chunk")
    parser.add_argument("--name", default="chunk2_LF_relu_hybrid_snn_ft_t4")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint yok: {args.checkpoint}")
    if not args.data.exists():
        raise FileNotFoundError(f"Data YAML yok: {args.data}")

    print("Building hybrid model")
    yolo, replaced, thresholds = build_hybrid_model(args, train_timesteps=args.train_timesteps)
    threshold_mean = sum(thresholds) / len(thresholds) if thresholds else args.threshold
    print(f"Replaced activations: {replaced}")
    print(f"Threshold mean/min/max: {threshold_mean:.6f}/{min(thresholds):.6f}/{max(thresholds):.6f}")

    yolo.model.timesteps = args.eval_timesteps
    pre_val = evaluate_yolo(yolo, args.data, "val", args.imgsz, args.batch, args.device)
    pre_test = evaluate_yolo(yolo, args.data, "test", args.imgsz, args.batch, args.device)

    print("\nRebuilding hybrid model for training after pre-eval")
    yolo, replaced, thresholds = build_hybrid_model(args, train_timesteps=args.train_timesteps)
    yolo.model.timesteps = args.train_timesteps
    train_overrides = augmentation_kwargs(args.augmentation)
    print(
        f"\nFine-tuning hybrid SNN: epochs={args.epochs}, train_T={args.train_timesteps}, "
        f"eval_T={args.eval_timesteps}, lr0={args.lr0}, augmentation={args.augmentation}"
    )
    yolo.train(
        trainer=trainer_for(yolo.model),
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        plots=False,
        patience=max(args.epochs, 20),
        lr0=args.lr0,
        lrf=1.0,
        warmup_epochs=0.0,
        optimizer="AdamW",
        **train_overrides,
    )

    yolo.model.timesteps = args.eval_timesteps
    post_val = evaluate_yolo(yolo, args.data, "val", args.imgsz, args.batch, args.device)
    post_test = evaluate_yolo(yolo, args.data, "test", args.imgsz, args.batch, args.device)

    row = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "scope": args.scope,
        "threshold": args.threshold,
        "source_activation": args.source_activation,
        "spike_scale": args.spike_scale,
        "calibration_stat": args.calibration_stat,
        "calibration_granularity": args.calibration_granularity,
        "surrogate_alpha": args.surrogate_alpha,
        "learn_thresholds": args.learn_thresholds,
        "replaced": replaced,
        "threshold_mean": round(threshold_mean, 6),
        "threshold_min": round(min(thresholds), 6) if thresholds else args.threshold,
        "threshold_max": round(max(thresholds), 6) if thresholds else args.threshold,
        "train_timesteps": args.train_timesteps,
        "eval_timesteps": args.eval_timesteps,
        "epochs": args.epochs,
        "lr0": args.lr0,
        "augmentation": args.augmentation,
        "run_dir": str(args.project / args.name),
        **metrics_row(pre_val, "pre_val"),
        **metrics_row(pre_test, "pre_test"),
        **metrics_row(post_val, "post_val"),
        **metrics_row(post_test, "post_test"),
    }
    write_result(row, args.out)
    print(f"\nSaved: {args.out}")
    print(
        f"pre_test mAP50={row['pre_test_map50']:.4f} mAP50-95={row['pre_test_map50_95']:.4f} | "
        f"post_test mAP50={row['post_test_map50']:.4f} mAP50-95={row['post_test_map50_95']:.4f}"
    )


if __name__ == "__main__":
    main()
