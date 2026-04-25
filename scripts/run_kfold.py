#!/usr/bin/env python3
"""
run_kfold.py — SubPipe 5-fold cross validation eğitim scripti

Önkoşul:
  - convert_pbm_to_png.py çalıştırılmış (.png'ler hazır)
  - prepare_kfold.py çalıştırılmış (subpipe_kfold/ klasörü hazır)

Her fold × modalite için:
  1. YOLO26n'i sıfırdan eğitir (50 epoch, imgsz=640, early stopping kapalı)
  2. best.pt ile val ve test metriklerini ayrı ayrı çeker
  3. results/kfold_results.csv'ye satır satır yazar (crash-safe)

Toplam: 5 fold × 2 modalite = 10 eğitim

Kullanım:
  python3 run_kfold.py                  # tümünü çalıştır
  python3 run_kfold.py --folds 0 1      # sadece fold 0 ve 1
  python3 run_kfold.py --mod LF         # sadece LF modalitesi
  python3 run_kfold.py --folds 0 --mod LF  # tek fold + tek mod (sanity check)
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────
MODEL    = "yolo26n.pt"
EPOCHS   = 50
IMGSZ    = 640
BATCH    = 16
WORKERS  = 8
PATIENCE = 100       # EPOCHS'tan büyük → early stopping devre dışı
DEVICE   = 0

KFOLD_DIR   = Path("/home/alp/thesis/datasets/subpipe_kfold")
RUNS_DIR    = Path("/home/alp/thesis/runs/kfold")
RESULTS_CSV = Path("/home/alp/thesis/results/kfold_results.csv")

CHUNKS     = [f"Chunk{i}" for i in range(5)]
MODALITIES = ["LF", "HF"]

CSV_FIELDS = [
    "fold", "modality", "test_chunk",
    "train_imgs", "val_imgs", "test_imgs",
    "val_precision", "val_recall", "val_map50", "val_map50_95",
    "test_precision", "test_recall", "test_map50", "test_map50_95",
    "best_epoch", "train_time_min", "run_dir",
]
# ───────────────────────────────────────────────────────────────────────────


def count_images(fold_dir: Path, modality: str, split: str) -> int:
    return len(list((fold_dir / modality / "images" / split).glob("*.png")))


def extract_metrics(results) -> dict:
    """Ultralytics val sonucundan metrikleri çek (versiyon-bağımsız)."""
    return {
        "precision": round(float(results.box.mp), 4),
        "recall":    round(float(results.box.mr), 4),
        "map50":     round(float(results.box.map50), 4),
        "map50_95":  round(float(results.box.map), 4),
    }


def get_best_epoch(run_dir: Path) -> int:
    """results.csv'den en yüksek val mAP50'nin epoch'unu döndür."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return -1
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): v for k, v in row.items()} for row in reader]
    col = "metrics/mAP50(B)"
    if not rows or col not in rows[0]:
        return -1
    best = max(rows, key=lambda r: float(r[col]))
    return int(float(best.get("epoch", -1)))


def already_done(fold_idx: int, modality: str) -> bool:
    """CSV'de bu fold+modalite zaten var mı?"""
    if not RESULTS_CSV.exists():
        return False
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            if int(row["fold"]) == fold_idx and row["modality"] == modality:
                return True
    return False


def write_csv_row(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def train_and_eval(fold_idx: int, modality: str, job_no: int, total: int) -> dict:
    test_chunk = CHUNKS[fold_idx]
    fold_dir   = KFOLD_DIR / f"fold_{fold_idx}"
    data_yaml  = fold_dir / modality / "data.yaml"
    run_name   = f"fold{fold_idx}_{modality}"

    print(f"\n{'='*64}")
    print(f"  [{job_no}/{total}]  Fold {fold_idx} · {modality}  "
          f"(test={test_chunk})   {datetime.now().strftime('%H:%M')}")
    print(f"{'='*64}")

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"YAML bulunamadı: {data_yaml}\n"
            f"prepare_kfold.py çalıştırıldı mı?"
        )

    t0 = time.time()

    # ── Eğitim ──────────────────────────────────────────────────────────────
    model = YOLO(MODEL)
    model.train(
        data     = str(data_yaml),
        epochs   = EPOCHS,
        imgsz    = IMGSZ,
        batch    = BATCH,
        workers  = WORKERS,
        patience = PATIENCE,
        device   = DEVICE,
        project  = str(RUNS_DIR),
        name     = run_name,
        exist_ok = True,
        plots    = True,
    )

    train_time = (time.time() - t0) / 60
    run_dir    = RUNS_DIR / run_name

    # ── Val + Test değerlendirmesi (best.pt ile) ─────────────────────────────
    best_pt    = run_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))

    val_r  = best_model.val(
        data=str(data_yaml), split="val",
        imgsz=IMGSZ, batch=BATCH, device=DEVICE, verbose=False,
    )
    test_r = best_model.val(
        data=str(data_yaml), split="test",
        imgsz=IMGSZ, batch=BATCH, device=DEVICE, verbose=False,
    )

    vm = extract_metrics(val_r)
    tm = extract_metrics(test_r)

    print(f"\n  Val  → P={vm['precision']:.3f}  R={vm['recall']:.3f}  "
          f"mAP50={vm['map50']:.4f}  mAP50-95={vm['map50_95']:.4f}")
    print(f"  Test → P={tm['precision']:.3f}  R={tm['recall']:.3f}  "
          f"mAP50={tm['map50']:.4f}  mAP50-95={tm['map50_95']:.4f}")
    print(f"  Süre : {train_time:.1f} dk")

    row = {
        "fold":           fold_idx,
        "modality":       modality,
        "test_chunk":     test_chunk,
        "train_imgs":     count_images(fold_dir, modality, "train"),
        "val_imgs":       count_images(fold_dir, modality, "val"),
        "test_imgs":      count_images(fold_dir, modality, "test"),
        "val_precision":  vm["precision"],
        "val_recall":     vm["recall"],
        "val_map50":      vm["map50"],
        "val_map50_95":   vm["map50_95"],
        "test_precision": tm["precision"],
        "test_recall":    tm["recall"],
        "test_map50":     tm["map50"],
        "test_map50_95":  tm["map50_95"],
        "best_epoch":     get_best_epoch(run_dir),
        "train_time_min": round(train_time, 1),
        "run_dir":        str(run_dir),
    }
    write_csv_row(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="SubPipe K-Fold eğitim")
    parser.add_argument("--folds", type=int, nargs="+",
                        default=list(range(5)),
                        help="Çalıştırılacak fold'lar (default: 0-4)")
    parser.add_argument("--mod", choices=["LF", "HF", "both"], default="both",
                        help="Modalite (default: both)")
    args = parser.parse_args()

    mods = MODALITIES if args.mod == "both" else [args.mod]
    jobs = [(f, m) for f in args.folds for m in mods]
    total = len(jobs)

    print("SubPipe 5-Fold K-Fold Eğitim")
    print(f"  Model     : {MODEL}  epochs={EPOCHS}  imgsz={IMGSZ}  batch={BATCH}")
    print(f"  İş sayısı : {total}  ({len(args.folds)} fold × {len(mods)} mod)")
    print(f"  Çıktı     : {RESULTS_CSV}")
    print(f"  Başlangıç : {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    pending = [(f, m) for f, m in jobs if not already_done(f, m)]
    skipped = total - len(pending)
    if skipped:
        print(f"  Atlanan   : {skipped} (CSV'de zaten mevcut)")

    total_start = time.time()
    completed: list[dict] = []
    failed: list[tuple[int, str]] = []

    for job_no, (fold_idx, modality) in enumerate(pending, start=skipped + 1):
        try:
            row = train_and_eval(fold_idx, modality, job_no, total)
            completed.append(row)
        except Exception as e:
            print(f"\n[HATA] Fold {fold_idx} {modality}: {e}")
            import traceback
            traceback.print_exc()
            failed.append((fold_idx, modality))
            print("Sonraki eğitime devam ediliyor...\n")

    total_min = (time.time() - total_start) / 60

    print(f"\n{'='*64}")
    print(f"Tamamlandı  {len(completed)} eğitim  |  "
          f"{len(failed)} hata  |  {total_min:.0f} dk toplam")
    if failed:
        print(f"  Başarısız: {failed}")
    print(f"Sonuçlar → {RESULTS_CSV}")

    if completed:
        print(f"\n{'Fold':>4}  {'Mod':>4}  {'Test':>8}  "
              f"{'val_mAP50':>9}  {'val_m50-95':>10}  "
              f"{'test_mAP50':>10}  {'test_m50-95':>11}")
        print("─" * 64)
        for r in completed:
            print(f"  {r['fold']:>2}   {r['modality']:<4}  {r['test_chunk']:<8}  "
                  f"{r['val_map50']:>9.4f}  {r['val_map50_95']:>10.4f}  "
                  f"{r['test_map50']:>10.4f}  {r['test_map50_95']:>11.4f}")


if __name__ == "__main__":
    main()