#!/usr/bin/env python3
"""
kfold_analysis.py — K-fold sonuçlarını analiz eder ve markdown rapor üretir.

Girdi : results/kfold_results.csv  (run_kfold.py'nin ürettiği)
Çıktı :
  - notes/kfold_analysis.md  (tezde kullanılabilir markdown rapor)
  - results/kfold_summary.csv (mean/std özet, makinece okunabilir)
  - Konsola yazdırılır

Üretilen tablolar:
  1. Fold-by-fold detay (her fold × modalite)
  2. Modalite başına özet (mean ± std)
  3. Modalite karşılaştırması (LF vs HF)
"""

import csv
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

# ── Config ─────────────────────────────────────────────────────────────────
RESULTS_CSV = Path("/home/alp/thesis/results/kfold_results.csv")
SUMMARY_CSV = Path("/home/alp/thesis/results/kfold_summary.csv")
OUTPUT_MD   = Path("/home/alp/thesis/notes/kfold_analysis.md")

METRIC_KEYS = [
    ("test_map50",     "Test mAP50"),
    ("test_map50_95",  "Test mAP50-95"),
    ("test_precision", "Test Precision"),
    ("test_recall",    "Test Recall"),
    ("val_map50",      "Val mAP50"),
    ("val_map50_95",   "Val mAP50-95"),
]
# ───────────────────────────────────────────────────────────────────────────


def load_rows() -> list[dict]:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{RESULTS_CSV} bulunamadı. run_kfold.py çalıştırıldı mı?"
        )
    with open(RESULTS_CSV) as f:
        return list(csv.DictReader(f))


def to_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def section_fold_detail(rows: list[dict]) -> list[str]:
    lines = ["\n## 1. Fold-by-Fold Sonuçlar\n"]

    for mod in ["LF", "HF"]:
        mod_rows = [r for r in rows if r["modality"] == mod]
        mod_rows.sort(key=lambda r: int(r["fold"]))
        if not mod_rows:
            continue

        lines.append(f"\n### {mod}\n")
        lines.append("| Fold | Test Chunk | Train img | Test img | "
                     "val mAP50 | val mAP50-95 | "
                     "test mAP50 | test mAP50-95 | "
                     "test P | test R | best epoch | süre (dk) |")
        lines.append("|------|------------|-----------|----------|"
                     "-----------|---------------|"
                     "------------|----------------|"
                     "--------|--------|-------------|------------|")

        for r in mod_rows:
            lines.append(
                f"| {r['fold']} | {r['test_chunk']} | {r['train_imgs']} | "
                f"{r['test_imgs']} | "
                f"{to_float(r['val_map50']):.4f} | "
                f"{to_float(r['val_map50_95']):.4f} | "
                f"{to_float(r['test_map50']):.4f} | "
                f"{to_float(r['test_map50_95']):.4f} | "
                f"{to_float(r['test_precision']):.3f} | "
                f"{to_float(r['test_recall']):.3f} | "
                f"{r['best_epoch']} | {r['train_time_min']} |"
            )

    return lines


def section_modality_summary(rows: list[dict]) -> tuple[list[str], list[dict]]:
    lines = ["\n## 2. Modalite Başına Özet (mean ± std)\n"]
    lines.append("| Modalite | n |" + "".join(f" {label} |" for _, label in METRIC_KEYS))
    lines.append("|----------|---|" + "|".join(["-" * 16] * len(METRIC_KEYS)) + "|")

    summary_rows: list[dict] = []

    for mod in ["LF", "HF"]:
        mod_rows = [r for r in rows if r["modality"] == mod]
        if not mod_rows:
            continue

        n = len(mod_rows)
        cells = [f"| **{mod}** | {n} |"]
        summary = {"modality": mod, "n_folds": n}

        for key, _ in METRIC_KEYS:
            values = [to_float(r[key]) for r in mod_rows]
            m, s = stats(values)
            cells.append(f" {m:.4f} ± {s:.4f} |")
            summary[f"{key}_mean"] = round(m, 4)
            summary[f"{key}_std"]  = round(s, 4)

        lines.append("".join(cells))
        summary_rows.append(summary)

    return lines, summary_rows


def section_comparison(summary_rows: list[dict]) -> list[str]:
    lines = ["\n## 3. LF vs HF Karşılaştırması (Test seti üzerinden)\n"]

    if len(summary_rows) < 2:
        lines.append("\n_Karşılaştırma için her iki modalite de gerekli._")
        return lines

    lf = next((r for r in summary_rows if r["modality"] == "LF"), None)
    hf = next((r for r in summary_rows if r["modality"] == "HF"), None)
    if lf is None or hf is None:
        return lines

    lines.append("| Metrik | LF | HF | Δ (HF − LF) |")
    lines.append("|--------|------|------|-------------|")

    for key, label in METRIC_KEYS:
        if not key.startswith("test_"):
            continue
        lf_m, hf_m = lf[f"{key}_mean"], hf[f"{key}_mean"]
        delta = hf_m - lf_m
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {label} | {lf_m:.4f} | {hf_m:.4f} | {sign}{delta:.4f} |")

    return lines


def section_robust_chunk(rows: list[dict]) -> list[str]:
    """Chunk3 (h_mean=0.45 anomalisi) test olduğunda nasıl performe ediyor?"""
    lines = ["\n## 4. Chunk3 Anomalisi (her fold ayrı analiz)\n"]
    lines.append(
        "_Dataset analizinde Chunk3'te boru kutu yüksekliğinin diğer "
        "chunk'ların yarısı olduğu (h≈0.45 vs ~0.91) tespit edildi. "
        "Aşağıda Chunk3 test seti olduğunda sonuçlar var._\n"
    )

    chunk3_rows = [r for r in rows if r["test_chunk"] == "Chunk3"]
    if not chunk3_rows:
        lines.append("\n_Henüz Chunk3 test fold'u tamamlanmadı._")
        return lines

    lines.append("| Modalite | test mAP50 | test mAP50-95 | P | R |")
    lines.append("|----------|------------|---------------|------|------|")
    for r in sorted(chunk3_rows, key=lambda r: r["modality"]):
        lines.append(
            f"| {r['modality']} | {to_float(r['test_map50']):.4f} | "
            f"{to_float(r['test_map50_95']):.4f} | "
            f"{to_float(r['test_precision']):.3f} | "
            f"{to_float(r['test_recall']):.3f} |"
        )

    return lines


def write_summary_csv(summary_rows: list[dict]) -> None:
    if not summary_rows:
        return
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    rows = load_rows()
    if not rows:
        print(f"{RESULTS_CSV} boş, hiç sonuç yok.")
        return

    n_total = len(rows)
    folds   = sorted({int(r["fold"]) for r in rows})
    mods    = sorted({r["modality"] for r in rows})

    out: list[str] = []
    out.append("# SubPipe K-Fold Cross Validation Sonuçları")
    out.append(f"\n_Oluşturulma: {datetime.now():%Y-%m-%d %H:%M}_")
    out.append(f"\nKaynak: `{RESULTS_CSV}`")
    out.append(f"\n**Tamamlanan eğitim:** {n_total}  |  **Fold'lar:** {folds}  |  "
               f"**Modaliteler:** {mods}")

    out += section_fold_detail(rows)
    summary_lines, summary_rows = section_modality_summary(rows)
    out += summary_lines
    out += section_comparison(summary_rows)
    out += section_robust_chunk(rows)

    out.append("\n---")
    out.append("\n_Not: Chunk3 test fold'unda anotasyon sayısı çok düşük (~30) "
               "olduğundan rakamlar diğer fold'lara kıyasla daha gürültülüdür. "
               "Standart sapma değerleri yorumlanırken bu göz önünde bulundurulmalıdır._")

    report = "\n".join(out)
    print(report)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(report)
    print(f"\n[Markdown rapor: {OUTPUT_MD}]")

    write_summary_csv(summary_rows)
    print(f"[Özet CSV: {SUMMARY_CSV}]")


if __name__ == "__main__":
    main()