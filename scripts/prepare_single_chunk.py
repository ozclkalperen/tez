#!/usr/bin/env python3
"""
prepare_single_chunk.py — Tek chunk/small-run YOLO dataset hazırlar.

Bu script mevcut küçük denemeler içindir. `datasets/SubPipe` flat yapıdaysa
doğrudan onu kullanır:

  datasets/SubPipe/
    SSS_LF_images/{Image,YOLO_Annotation}
    SSS_HF_images/{Image,YOLO_Annotation}

Tam veri seti geldiğinde belirli bir chunk için de kullanılabilir:

  python3 scripts/prepare_single_chunk.py --chunk Chunk2

Çıktı:
  datasets/subpipe_single_chunk/{LF,HF}/
    images/{train,val,test}
    labels/{train,val,test}
    data.yaml
"""

import argparse
import os
import random
from pathlib import Path

import yaml

from paths import DATASETS_DIR, REPO_ROOT, get_subpipe_root


SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
MODALITIES = ["LF", "HF"]
MODALITY_DIR = {"LF": "SSS_LF_images", "HF": "SSS_HF_images"}
OUT_DIR = DATASETS_DIR / "subpipe_single_chunk"


def modality_root(subpipe_root: Path, modality: str, chunk: str | None) -> Path:
    if chunk:
        return subpipe_root / chunk / MODALITY_DIR[modality]
    return subpipe_root / MODALITY_DIR[modality]


def get_images(root: Path) -> list[Path]:
    img_dir = root / "Image"
    pngs = sorted(img_dir.glob("*.png"))
    if pngs:
        return pngs
    return sorted(img_dir.glob("*.pbm"))


def label_for(img: Path) -> Path | None:
    lbl = img.parent.parent / "YOLO_Annotation" / f"{img.stem}.txt"
    return lbl if lbl.exists() else None


def make_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src.resolve(), start=dst.parent))


def write_yaml(mod_dir: Path) -> None:
    cfg = {
        "path": str(mod_dir.relative_to(REPO_ROOT)),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["pipe"],
    }
    with open(mod_dir / "data.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def split_images(images: list[Path], seed: int) -> dict[str, list[Path]]:
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * TRAIN_FRACTION)
    n_val = round(n * VAL_FRACTION)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def prepare_modality(subpipe_root: Path, modality: str, chunk: str | None) -> None:
    src_root = modality_root(subpipe_root, modality, chunk)
    images = get_images(src_root)
    if not images:
        raise FileNotFoundError(f"{modality}: görüntü bulunamadı: {src_root / 'Image'}")

    mod_dir = OUT_DIR / modality
    splits = split_images(images, SEED + MODALITIES.index(modality))
    labeled_counts: dict[str, int] = {}

    for split, split_images_ in splits.items():
        img_dir = mod_dir / "images" / split
        lbl_dir = mod_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        labeled = 0
        for img in split_images_:
            make_symlink(img, img_dir / img.name)
            dst_lbl = lbl_dir / f"{img.stem}.txt"
            src_lbl = label_for(img)
            if src_lbl:
                make_symlink(src_lbl, dst_lbl)
                labeled += 1
            else:
                if dst_lbl.exists() or dst_lbl.is_symlink():
                    dst_lbl.unlink()
                dst_lbl.touch()
        labeled_counts[split] = labeled

    for cache in (mod_dir / "labels").glob("*.cache"):
        cache.unlink()

    write_yaml(mod_dir)

    print(f"\n{modality} → {mod_dir}")
    for split, split_images_ in splits.items():
        n = len(split_images_)
        labeled = labeled_counts[split]
        pct = labeled / n * 100 if n else 0
        ext = split_images_[0].suffix if split_images_ else ""
        print(f"  {split:5s}: {n:4d} img{ext:4s}  {labeled:4d} labeled ({pct:.0f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tek chunk small-run dataset hazırlar")
    parser.add_argument(
        "--chunk",
        default=None,
        help="Tam SubPipe yapısında kullanılacak chunk adı, örn. Chunk2. Flat yapı için boş bırakın.",
    )
    args = parser.parse_args()

    subpipe_root = get_subpipe_root(require_full=bool(args.chunk))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("SubPipe single-chunk dataset hazırlığı")
    print(f"  Kaynak : {subpipe_root if args.chunk is None else subpipe_root / args.chunk}")
    print(f"  Çıktı  : {OUT_DIR}")
    print(f"  Split  : train={TRAIN_FRACTION:.0%}, val={VAL_FRACTION:.0%}, test={1 - TRAIN_FRACTION - VAL_FRACTION:.0%}")

    for modality in MODALITIES:
        prepare_modality(subpipe_root, modality, args.chunk)

    print("\nTamamlandı.")


if __name__ == "__main__":
    main()
