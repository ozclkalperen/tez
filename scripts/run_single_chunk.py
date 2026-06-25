#!/usr/bin/env python3
"""
run_single_chunk.py — Tek chunk üstünde küçük YOLO deneyi.

Bu script SNN denemelerine geçmeden önce hızlı bir ANN checkpoint üretmek için
tasarlandı. Tam 5-fold protokolünü etkilemez.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from torch import nn
from ultralytics import YOLO

from paths import DATASETS_DIR, REPO_ROOT, RESULTS_DIR as RESULTS_ROOT, RUNS_DIR as RUNS_ROOT


MODEL = str(REPO_ROOT / "yolo26n.pt") if (REPO_ROOT / "yolo26n.pt").exists() else "yolo26n.pt"
DEFAULT_DATASET_DIR = DATASETS_DIR / "subpipe_single_chunk"
DEFAULT_RUN_GROUP = "single_chunk"
DEFAULT_TAG = "chunk2"
MODALITIES = ["LF", "HF"]


def replace_activation(model, activation: str) -> int:
    if activation == "silu":
        return 0
    if activation != "relu":
        raise ValueError(f"Bilinmeyen activation: {activation}")

    replaced = 0
    for module in model.modules():
        if hasattr(module, "act") and type(module.act).__name__ == "SiLU":
            module.act = nn.ReLU(inplace=True)
            replaced += 1
    return replaced


def build_training_model(activation: str, init_dir: Path) -> tuple[YOLO, int, Path | None]:
    """Return a YOLO object whose checkpoint really contains requested activations."""
    model = YOLO(MODEL)
    replaced = replace_activation(model.model, activation)

    if activation == "relu":
        model.model.yaml["activation"] = "torch.nn.ReLU(inplace=True)"
        init_dir.mkdir(parents=True, exist_ok=True)
        init_pt = init_dir / "yolo26n_relu.pt"
        model.save(str(init_pt))
        model = YOLO(str(init_pt))
        return model, replaced, init_pt

    return model, replaced, None


def extract_metrics(results) -> dict[str, float]:
    return {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
    }


def run(
    modality: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    activation: str,
    dataset_dir: Path,
    run_group: str,
    tag: str,
) -> None:
    runs_dir = RUNS_ROOT / run_group
    results_dir = RESULTS_ROOT / run_group
    init_dir = runs_dir / "init"

    data_yaml = dataset_dir / modality / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML yok: {data_yaml}. Önce prepare_single_chunk.py çalıştırın.")

    suffix = "" if activation == "silu" else f"_{activation}"
    run_name = f"{tag}_{modality}{suffix}"
    model, replaced, init_pt = build_training_model(activation, init_dir)
    if replaced:
        print(f"{replaced} SiLU activation ReLU ile değiştirildi ve başlangıç checkpoint'i kaydedildi: {init_pt}")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(runs_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
        patience=max(epochs, 20),
    )

    best_pt = runs_dir / run_name / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))
    val_r = best_model.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=batch, device=device, verbose=False)
    test_r = best_model.val(data=str(data_yaml), split="test", imgsz=imgsz, batch=batch, device=device, verbose=False)

    vm = extract_metrics(val_r)
    tm = extract_metrics(test_r)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{run_name}_metrics.txt"
    out.write_text(
        "\n".join(
            [
                f"modality={modality}",
                f"activation={activation}",
                f"dataset_dir={dataset_dir}",
                f"run_group={run_group}",
                f"tag={tag}",
                f"replaced_activations={replaced}",
                f"init_pt={init_pt if init_pt else MODEL}",
                f"run_dir={runs_dir / run_name}",
                f"best_pt={best_pt}",
                f"val_precision={vm['precision']:.4f}",
                f"val_recall={vm['recall']:.4f}",
                f"val_map50={vm['map50']:.4f}",
                f"val_map50_95={vm['map50_95']:.4f}",
                f"test_precision={tm['precision']:.4f}",
                f"test_recall={tm['recall']:.4f}",
                f"test_map50={tm['map50']:.4f}",
                f"test_map50_95={tm['map50_95']:.4f}",
            ]
        )
        + "\n"
    )
    print(out.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Tek Chunk2 küçük ANN deneyi")
    parser.add_argument("--mod", choices=["LF", "HF", "both"], default="LF")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--activation", choices=["silu", "relu"], default="silu")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="LF/HF data.yaml klasörlerini içeren dataset kökü.",
    )
    parser.add_argument(
        "--run-group",
        default=DEFAULT_RUN_GROUP,
        help="runs/ ve results/ altında kullanılacak grup adı.",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="Run adı öneki. Örn: chunk2, chunk4, pilot_chunk2.",
    )
    args = parser.parse_args()

    mods = MODALITIES if args.mod == "both" else [args.mod]
    for modality in mods:
        run(
            modality=modality,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            activation=args.activation,
            dataset_dir=args.dataset_dir,
            run_group=args.run_group,
            tag=args.tag,
        )


if __name__ == "__main__":
    main()
