#!/usr/bin/env python3
"""
snn_probe.py — YOLO checkpoint'i SpikingJelly ANN-to-SNN için yoklar.

Bu script gerçek SNN sonucundan önce bir uygunluk/başlangıç denemesidir:
  - torch, ultralytics, spikingjelly importlarını kontrol eder
  - checkpoint içindeki modül tiplerini özetler
  - ANN-to-SNN converter çağrısını dener

YOLO26 SiLU/C2f benzeri ReLU-dışı bloklar içerdiği için conversion doğrudan
çalışmayabilir. Hata mesajı bir sonraki uyarlama adımını gösterir.
"""

import argparse
from collections import Counter
from pathlib import Path

from paths import REPO_ROOT, RUNS_DIR


def default_checkpoint(modality: str) -> Path:
    single = RUNS_DIR / "single_chunk" / f"chunk2_{modality}" / "weights" / "best.pt"
    if single.exists():
        return single
    kfold = RUNS_DIR / "kfold" / f"fold2_{modality}" / "weights" / "best.pt"
    if kfold.exists():
        return kfold
    return REPO_ROOT / "yolo26n.pt"


def require_imports():
    try:
        import torch
        from ultralytics import YOLO
        from spikingjelly.activation_based import ann2snn
    except Exception as exc:
        raise SystemExit(
            "Gerekli SNN bağımlılıkları eksik veya import edilemiyor.\n"
            "Gerekli paketler: torch, ultralytics, spikingjelly.\n"
            f"Import hatası: {type(exc).__name__}: {exc}"
        ) from exc
    return torch, YOLO, ann2snn


def summarize_modules(model) -> None:
    counts = Counter(type(module).__name__ for module in model.modules())
    print("\nEn sık modül tipleri:")
    for name, count in counts.most_common(25):
        print(f"  {name:24s} {count}")

    suspicious = {
        name: count
        for name, count in counts.items()
        if name.lower() in {"silu", "swish", "gelu", "mish", "hardswish"}
    }
    if suspicious:
        print("\nDikkat: ReLU-dışı aktivasyonlar bulundu:")
        for name, count in sorted(suspicious.items()):
            print(f"  {name}: {count}")
        print("SpikingJelly ann2snn dönüşümü bu katmanlarda doğrudan takılabilir.")


def try_convert(ann2snn, model) -> None:
    converter = ann2snn.Converter(mode="max", dataloader=None)
    snn_model = converter(model)
    print("\nDönüşüm başarılı.")
    print(snn_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="SpikingJelly ANN-to-SNN uygunluk probu")
    parser.add_argument("--mod", choices=["LF", "HF"], default="LF")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--convert", action="store_true", help="ann2snn converter çağrısını dene")
    args = parser.parse_args()

    torch, YOLO, ann2snn = require_imports()
    ckpt = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.mod)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt}")

    print(f"torch={torch.__version__}")
    print(f"checkpoint={ckpt}")

    yolo = YOLO(str(ckpt))
    model = yolo.model.eval()
    summarize_modules(model)

    if args.convert:
        try_convert(ann2snn, model)
    else:
        print("\nConverter çağrısı yapılmadı. Denemek için `--convert` ekleyin.")


if __name__ == "__main__":
    main()
