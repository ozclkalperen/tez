#!/usr/bin/env python3
"""
run_hybrid_snn.py — YOLO checkpoint üzerinde hybrid SNN inference denemesi.

Standart SpikingJelly ann2snn converter, Ultralytics YOLO26 forward akışını
torch.fx ile trace edemediği için bu script daha basit bir ara adım dener:

  1. YOLO checkpoint yüklenir.
  2. Seçilen Conv.act kaynak aktivasyonları IFNode ile değiştirilir.
  3. Aynı görüntü T timestep boyunca forward edilir.
  4. Detection output ortalaması Ultralytics validator ile değerlendirilir.

Bu tam SNN dönüşümü değildir; hybrid smoke/evaluation aracıdır.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import cv2
import torch
import yaml
from torch import nn
from ultralytics import YOLO
from spikingjelly.activation_based import base as sj_base
from spikingjelly.activation_based import functional, neuron, surrogate

from paths import DATASETS_DIR, RESULTS_DIR, RUNS_DIR


DEFAULT_DATA = DATASETS_DIR / "subpipe_single_chunk" / "LF" / "data.yaml"
DEFAULT_CHECKPOINT = RUNS_DIR / "single_chunk" / "chunk2_LF_relu" / "weights" / "best.pt"
OUT_CSV = RESULTS_DIR / "single_chunk" / "hybrid_snn_sweep.csv"
STATS_CSV = RESULTS_DIR / "single_chunk" / "activation_stats.csv"


class ScaledIFNode(nn.Module):
    def __init__(self, threshold: float, output_scale: float) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.output_scale = float(output_scale)
        self.node = neuron.IFNode(v_threshold=self.threshold, v_reset=None, detach_reset=True)

    def forward(self, x):
        return self.node(x) * self.output_scale


class ChannelScaledIFNode(sj_base.MemoryModule):
    def __init__(
        self,
        thresholds: torch.Tensor,
        spike_scale: str,
        surrogate_alpha: float = 4.0,
        learn_thresholds: bool = False,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        if thresholds.ndim != 1:
            raise ValueError("Channel thresholds must be a 1D tensor")
        thresholds = thresholds.float().clamp_min(1e-6)
        threshold_view = thresholds.view(1, -1, 1, 1)
        if learn_thresholds:
            self.thresholds = nn.Parameter(threshold_view.clone())
        else:
            self.register_buffer("thresholds", threshold_view)
        self.spike_scale = spike_scale
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate.Sigmoid(alpha=surrogate_alpha, spiking=True)
        self.register_memory("v", None)

    def forward(self, x):
        thresholds = self.thresholds.clamp_min(1e-6).to(device=x.device, dtype=x.dtype)
        output_scale = thresholds if self.spike_scale == "threshold" else torch.ones_like(thresholds)
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        self.v = self.v + x
        spike = self.surrogate_function(self.v - thresholds)
        reset_spike = spike.detach() if self.detach_reset else spike
        self.v = self.v - reset_spike * thresholds
        return spike * output_scale


class HybridSNNModel(nn.Module):
    def __init__(self, base: nn.Module, timesteps: int) -> None:
        super().__init__()
        self.base = base
        self.timesteps = timesteps
        self.pt = True
        for attr in ["stride", "names", "end2end", "yaml", "args", "task", "pt_path"]:
            if hasattr(base, attr):
                setattr(self, attr, getattr(base, attr))

    def set_head_attr(self, *args, **kwargs):
        return self.base.set_head_attr(*args, **kwargs)

    def fuse(self, *args, **kwargs):
        self.base.fuse(*args, **kwargs)
        return self

    @property
    def criterion(self):
        return getattr(self.base, "criterion", None)

    @criterion.setter
    def criterion(self, value):
        setattr(self.base, "criterion", value)

    def loss(self, batch, preds=None):
        for attr in ["nc", "names", "args"]:
            if hasattr(self, attr):
                setattr(self.base, attr, getattr(self, attr))
        if preds is None:
            preds = self.forward(batch["img"])
        return self.base.loss(batch, preds)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)

        acc = None
        last_aux = None
        functional.reset_net(self.base)
        for _ in range(self.timesteps):
            out = self.base(x, *args, **kwargs)
            pred = out[0] if isinstance(out, tuple) else out
            last_aux = out[1:] if isinstance(out, tuple) else None
            acc = pred if acc is None else add_outputs(acc, pred)

        pred = scale_output(acc, 1.0 / self.timesteps)
        if last_aux is not None:
            return (pred, *last_aux)
        return pred


def add_outputs(left, right):
    if isinstance(left, torch.Tensor):
        return left + right
    if isinstance(left, dict):
        return {key: add_outputs(left[key], right[key]) for key in left}
    if isinstance(left, list):
        return [add_outputs(l_item, r_item) for l_item, r_item in zip(left, right)]
    if isinstance(left, tuple):
        return tuple(add_outputs(l_item, r_item) for l_item, r_item in zip(left, right))
    return left + right


def scale_output(output, factor: float):
    if isinstance(output, torch.Tensor):
        return output * factor
    if isinstance(output, dict):
        return {key: scale_output(value, factor) for key, value in output.items()}
    if isinstance(output, list):
        return [scale_output(item, factor) for item in output]
    if isinstance(output, tuple):
        return tuple(scale_output(item, factor) for item in output)
    return output * factor


def in_scope(module_name: str, scope: str) -> bool:
    if "+" in scope:
        return any(in_scope(module_name, part) for part in scope.split("+"))

    idx = top_level_index(module_name)
    if scope == "none":
        return False
    if scope == "all":
        return True
    if scope == "backbone":
        return not module_name.startswith("model.23")
    if scope == "head":
        return module_name.startswith("model.23")
    if scope.startswith("until"):
        limit = int(scope.replace("until", ""))
        return idx <= limit
    if scope.startswith("after"):
        start = int(scope.replace("after", ""))
        return idx > start
    if scope.startswith("only"):
        target = int(scope.replace("only", ""))
        return idx == target
    if scope.startswith("range"):
        raw = scope.replace("range", "")
        start, end = [int(item) for item in raw.split("-", maxsplit=1)]
        return start <= idx <= end
    raise ValueError(f"Bilinmeyen scope: {scope}")


def top_level_index(module_name: str) -> int:
    parts = module_name.split(".")
    if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return 999


def collect_activation_stats(
    checkpoint: Path,
    data_yaml: Path,
    source_activation: str,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    sample_limit: int,
    granularity: str = "global",
) -> dict[str, dict[str, float]]:
    yolo = YOLO(str(checkpoint))
    yolo.model.to(device).eval()
    sources = {"SiLU", "ReLU"} if source_activation == "both" else {source_activation}
    raw_stats: dict[str, dict] = {}
    hooks = []
    per_channel_limit = max(sample_limit // 256, 64)

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            output = output.detach().float()
            if output.numel() == 0:
                return
            values = output.flatten()
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                return
            state = raw_stats.setdefault(name, {"max": 0.0, "sum": 0.0, "count": 0, "samples": []})
            state["max"] = max(state["max"], float(values.max().item()))
            state["sum"] += float(values.double().sum().item())
            state["count"] += int(values.numel())

            remaining = sample_limit - sum(t.numel() for t in state["samples"])
            if remaining <= 0:
                return
            if values.numel() > remaining:
                step = max(values.numel() // remaining, 1)
                values = values[::step][:remaining]
            state["samples"].append(values.cpu())

            if granularity == "channel" and output.ndim >= 2:
                channel_values = output.movedim(1, 0).reshape(output.shape[1], -1)
                channel_values = channel_values.cpu()
                channel_state = state.setdefault("channel_samples", [[] for _ in range(output.shape[1])])
                channel_max = state.setdefault("channel_max", torch.zeros(output.shape[1]))
                if len(channel_state) != output.shape[1]:
                    return
                finite = torch.isfinite(channel_values)
                safe_values = torch.where(finite, channel_values, torch.zeros_like(channel_values))
                channel_max.copy_(torch.maximum(channel_max, safe_values.max(dim=1).values))
                for idx in range(output.shape[1]):
                    remaining_channel = per_channel_limit - sum(t.numel() for t in channel_state[idx])
                    if remaining_channel <= 0:
                        continue
                    vals = channel_values[idx][finite[idx]]
                    if vals.numel() == 0:
                        continue
                    if vals.numel() > remaining_channel:
                        step = max(vals.numel() // remaining_channel, 1)
                        vals = vals[::step][:remaining_channel]
                    channel_state[idx].append(vals)

        return hook

    for name, module in yolo.model.named_modules():
        if hasattr(module, "act") and type(module.act).__name__ in sources:
            hooks.append(module.register_forward_hook(make_hook(name)))

    image_paths = split_image_paths(data_yaml, split)
    try:
        for start in range(0, len(image_paths), batch):
            batch_paths = image_paths[start : start + batch]
            images = [load_image(path, imgsz) for path in batch_paths]
            x = torch.stack(images).to(device)
            with torch.no_grad():
                yolo.model(x)
    finally:
        for hook in hooks:
            hook.remove()

    stats: dict[str, dict[str, float]] = {}
    for name, state in raw_stats.items():
        samples = torch.cat(state["samples"]) if state["samples"] else torch.tensor([0.0])
        stats[name] = {
            "mean": state["sum"] / max(state["count"], 1),
            "max": state["max"],
            "p95": float(torch.quantile(samples, 0.95).item()),
            "p99": float(torch.quantile(samples, 0.99).item()),
            "p999": float(torch.quantile(samples, 0.999).item()),
        }
        if granularity == "channel" and "channel_samples" in state:
            channel_stats = {"p95": [], "p99": [], "p999": [], "max": []}
            for idx, channel_samples in enumerate(state["channel_samples"]):
                channel_tensor = torch.cat(channel_samples) if channel_samples else torch.tensor([0.0])
                channel_stats["p95"].append(float(torch.quantile(channel_tensor, 0.95).item()))
                channel_stats["p99"].append(float(torch.quantile(channel_tensor, 0.99).item()))
                channel_stats["p999"].append(float(torch.quantile(channel_tensor, 0.999).item()))
                channel_stats["max"].append(float(state["channel_max"][idx].item()))
            stats[name]["channel"] = channel_stats
    return stats


def split_image_paths(data_yaml: Path, split: str) -> list[Path]:
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute() and not base.exists():
        base = data_yaml.parent / base

    split_dir = Path(cfg[split])
    if not split_dir.is_absolute():
        split_dir = base / split_dir

    image_paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        image_paths.extend(split_dir.glob(pattern))

    if not image_paths:
        raise FileNotFoundError(f"Görüntü bulunamadı: {split_dir}")
    return sorted(image_paths)


def load_image(path: Path, imgsz: int) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    image = image[:, :, ::-1].copy()
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def write_activation_stats(stats: dict[str, dict[str, float]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["module", "mean", "p95", "p99", "p999", "max"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, values in stats.items():
            writer.writerow({"module": name, **{key: round(values[key], 6) for key in fields[1:]}})


def replacement_threshold(
    module_name: str,
    threshold: float,
    calibration_stat: str,
    calibration_stats: dict[str, dict[str, float]] | None,
) -> float:
    if calibration_stat == "none":
        return threshold
    if calibration_stats is None or module_name not in calibration_stats:
        return threshold
    return max(calibration_stats[module_name][calibration_stat] * threshold, 1e-6)


def replacement_channel_thresholds(
    module_name: str,
    threshold: float,
    calibration_stat: str,
    calibration_stats: dict[str, dict[str, float]] | None,
) -> torch.Tensor | None:
    if calibration_stat == "none" or calibration_stats is None or module_name not in calibration_stats:
        return None
    channel_stats = calibration_stats[module_name].get("channel")
    if not channel_stats or calibration_stat not in channel_stats:
        return None
    values = torch.tensor(channel_stats[calibration_stat], dtype=torch.float32)
    return (values * threshold).clamp_min(1e-6)


def replace_activations(
    base: nn.Module,
    scope: str,
    threshold: float,
    source_activation: str,
    spike_scale: str,
    calibration_stat: str,
    calibration_stats: dict[str, dict[str, float]] | None,
    calibration_granularity: str = "global",
    surrogate_alpha: float = 4.0,
    learn_thresholds: bool = False,
) -> tuple[int, list[float]]:
    replaced = 0
    thresholds = []
    sources = {"SiLU", "ReLU"} if source_activation == "both" else {source_activation}
    for name, module in base.named_modules():
        if not hasattr(module, "act") or type(module.act).__name__ not in sources:
            continue
        if not in_scope(name, scope):
            continue
        channel_thresholds = None
        if calibration_granularity == "channel":
            channel_thresholds = replacement_channel_thresholds(name, threshold, calibration_stat, calibration_stats)
        if channel_thresholds is not None:
            module.act = ChannelScaledIFNode(
                thresholds=channel_thresholds,
                spike_scale=spike_scale,
                surrogate_alpha=surrogate_alpha,
                learn_thresholds=learn_thresholds,
            )
            thresholds.extend(float(item) for item in channel_thresholds.tolist())
        else:
            effective_threshold = replacement_threshold(name, threshold, calibration_stat, calibration_stats)
            if learn_thresholds:
                module.act = ChannelScaledIFNode(
                    thresholds=torch.tensor([effective_threshold], dtype=torch.float32),
                    spike_scale=spike_scale,
                    surrogate_alpha=surrogate_alpha,
                    learn_thresholds=True,
                )
            else:
                output_scale = effective_threshold if spike_scale == "threshold" else 1.0
                module.act = ScaledIFNode(threshold=effective_threshold, output_scale=output_scale)
            thresholds.append(effective_threshold)
        replaced += 1
    return replaced, thresholds


def evaluate(
    checkpoint: Path,
    data_yaml: Path,
    scope: str,
    threshold: float,
    source_activation: str,
    timesteps: int,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    spike_scale: str,
    calibration_stat: str,
    calibration_stats: dict[str, dict[str, float]] | None,
    calibration_granularity: str = "global",
) -> dict:
    yolo = YOLO(str(checkpoint))
    base = copy.deepcopy(yolo.model).eval()
    replaced, replacement_thresholds = replace_activations(
        base=base,
        scope=scope,
        threshold=threshold,
        source_activation=source_activation,
        spike_scale=spike_scale,
        calibration_stat=calibration_stat,
        calibration_stats=calibration_stats,
        calibration_granularity=calibration_granularity,
    )

    if scope != "none":
        yolo.model = HybridSNNModel(base, timesteps=timesteps).eval()

    metrics = yolo.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        plots=False,
    )

    return {
        "scope": scope,
        "threshold": threshold,
        "source_activation": source_activation,
        "timesteps": timesteps,
        "replaced": replaced,
        "spike_scale": spike_scale,
        "calibration_stat": calibration_stat,
        "calibration_granularity": calibration_granularity,
        "threshold_mean": round(sum(replacement_thresholds) / len(replacement_thresholds), 6)
        if replacement_thresholds
        else threshold,
        "threshold_min": round(min(replacement_thresholds), 6) if replacement_thresholds else threshold,
        "threshold_max": round(max(replacement_thresholds), 6) if replacement_thresholds else threshold,
        "split": split,
        "precision": round(float(metrics.box.mp), 6),
        "recall": round(float(metrics.box.mr), 6),
        "map50": round(float(metrics.box.map50), 6),
        "map50_95": round(float(metrics.box.map), 6),
    }


def parse_csv_values(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid SNN YOLO evaluation")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scopes", default="none,until0,until5,backbone,all")
    parser.add_argument("--thresholds", default="0.1,0.5,1.0")
    parser.add_argument("--source-activation", choices=["SiLU", "ReLU", "both"], default="ReLU")
    parser.add_argument("--spike-scale", choices=["one", "threshold"], default="one")
    parser.add_argument("--calibration-stat", choices=["none", "p95", "p99", "p999", "max"], default="none")
    parser.add_argument("--calibration-granularity", choices=["global", "channel"], default="global")
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--calibration-samples", type=int, default=20000)
    parser.add_argument("--stats-out", type=Path, default=STATS_CSV)
    parser.add_argument("--timesteps", default="1,2,4")
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    scopes = parse_csv_values(args.scopes, str)
    thresholds = parse_csv_values(args.thresholds, float)
    timesteps = parse_csv_values(args.timesteps, int)

    calibration_stats = None
    if args.calibration_stat != "none":
        print(f"Collecting activation stats split={args.calibration_split} stat={args.calibration_stat}")
        calibration_stats = collect_activation_stats(
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
        write_activation_stats(calibration_stats, args.stats_out)
        print(f"Saved stats: {args.stats_out}")

    rows = []
    for scope in scopes:
        th_values = [1.0] if scope == "none" else thresholds
        t_values = [1] if scope == "none" else timesteps
        for threshold in th_values:
            for timestep in t_values:
                print(f"\nEvaluating scope={scope} threshold={threshold} T={timestep}")
                row = evaluate(
                    checkpoint=args.checkpoint,
                    data_yaml=args.data,
                    scope=scope,
                    threshold=threshold,
                    source_activation=args.source_activation,
                    timesteps=timestep,
                    split=args.split,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    spike_scale=args.spike_scale,
                    calibration_stat=args.calibration_stat,
                    calibration_stats=calibration_stats,
                    calibration_granularity=args.calibration_granularity,
                )
                rows.append(row)
                print(
                    f"  replaced={row['replaced']} "
                    f"mAP50={row['map50']:.4f} mAP50-95={row['map50_95']:.4f}"
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
