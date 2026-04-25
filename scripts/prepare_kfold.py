#!/usr/bin/env python3
"""
prepare_kfold.py — SubPipe 5-fold cross validation dataset hazırlığı

ÖNEMLİ: Önce convert_pbm_to_png.py çalıştırılmış olmalı.

Her fold: 1 chunk test, 4 chunk train/val havuzu.
Havuzdan random %10 val (best epoch seçimi), %90 train.
.png dosyalarını symlink'ler.

Çıktı yapısı:
  subpipe_kfold/
    fold_{i}/
      {LF,HF}/
        images/{train,val,test}/  ← .png symlinks
        labels/{train,val,test}/  ← .txt symlinks (veya boş .txt)
        data.yaml
"""

import random
from pathlib import Path
import yaml

# ── Config ─────────────────────────────────────────────────────────────────
SEED         = 42
VAL_FRACTION = 0.10
N_FOLDS      = 5
MODALITIES   = ["LF", "HF"]

BASE_DIR = Path("/home/alp/thesis/datasets/SubPipe")
OUT_DIR  = Path("/home/alp/thesis/datasets/subpipe_kfold")
CHUNKS   = [f"Chunk{i}" for i in range(N_FOLDS)]

MODALITY_DIR = {"LF": "SSS_LF_images", "HF": "SSS_HF_images"}
# ───────────────────────────────────────────────────────────────────────────


def get_images(chunk: str, modality: str) -> list[Path]:
    """O chunk/modalite içindeki tüm .png görüntüleri döndür."""
    img_dir = BASE_DIR / chunk / MODALITY_DIR[modality] / "Image"
    return sorted(img_dir.glob("*.png"))


def label_for(img: Path) -> Path | None:
    ann_dir = img.parent.parent / "YOLO_Annotation"
    lbl = ann_dir / (img.stem + ".txt")
    return lbl if lbl.exists() else None


def make_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def write_yaml(fold_dir: Path, modality: str) -> Path:
    cfg = {
        "path":  str(fold_dir / modality),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": ["pipe"],
    }
    yaml_path = fold_dir / modality / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def prepare_fold(fold_idx: int, test_chunk: str, train_chunks: list[str]) -> None:
    print(f"\n{'─'*60}")
    print(f"Fold {fold_idx}  test={test_chunk}  train={train_chunks}")
    print(f"{'─'*60}")

    fold_dir = OUT_DIR / f"fold_{fold_idx}"

    for modality in MODALITIES:
        test_images = get_images(test_chunk, modality)

        train_pool: list[Path] = []
        for ch in train_chunks:
            train_pool.extend(get_images(ch, modality))

        if not train_pool:
            print(f"  [HATA] {modality}: train_pool boş! "
                  f"convert_pbm_to_png.py çalıştırıldı mı?")
            continue

        rng = random.Random(SEED + fold_idx)
        rng.shuffle(train_pool)
        n_val        = max(1, round(len(train_pool) * VAL_FRACTION))
        val_images   = train_pool[:n_val]
        train_images = train_pool[n_val:]

        splits = {"train": train_images, "val": val_images, "test": test_images}
        n_labeled: dict[str, int] = {}

        for split, images in splits.items():
            img_dir = fold_dir / modality / "images" / split
            lbl_dir = fold_dir / modality / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            labeled = 0
            for img in images:
                make_symlink(img.resolve(), img_dir / img.name)

                dst_lbl = lbl_dir / (img.stem + ".txt")
                src_lbl = label_for(img)
                if src_lbl is not None:
                    make_symlink(src_lbl.resolve(), dst_lbl)
                    labeled += 1
                else:
                    if dst_lbl.exists() or dst_lbl.is_symlink():
                        dst_lbl.unlink()
                    dst_lbl.touch()

            n_labeled[split] = labeled

        yaml_path = write_yaml(fold_dir, modality)

        print(f"  {modality}:")
        for split, images in splits.items():
            pct = n_labeled[split] / len(images) * 100 if images else 0
            print(f"    {split:5s}: {len(images):4d} img  "
                  f"{n_labeled[split]:4d} labeled ({pct:.0f}%)")
        print(f"    YAML → {yaml_path.relative_to(OUT_DIR)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("SubPipe 5-Fold Cross Validation Hazırlığı")
    print(f"  Kaynak       : {BASE_DIR}")
    print(f"  Çıktı        : {OUT_DIR}")
    print(f"  Seed         : {SEED}")
    print(f"  Val fraction : {VAL_FRACTION:.0%}")
    print(f"  Chunk sırası : {CHUNKS}\n")

    for fold_idx in range(N_FOLDS):
        test_chunk   = CHUNKS[fold_idx]
        train_chunks = [c for c in CHUNKS if c != test_chunk]
        prepare_fold(fold_idx, test_chunk, train_chunks)

    print(f"\nTamamlandı — {N_FOLDS} fold × {len(MODALITIES)} modalite hazır.")
    print(f"Klasör: {OUT_DIR}\n")


if __name__ == "__main__":
    main()