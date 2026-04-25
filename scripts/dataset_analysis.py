#!/usr/bin/env python3
"""
dataset_analysis.py — SubPipe veri seti istatistik raporu.

Her chunk için:
  - Görüntü ve anotasyon sayıları
  - Anotasyon yoğunluğu
  - Boru pozisyon (cx, cy) ve boyut (w, h) istatistikleri

Çıktı:
  - Konsola yazılır
  - notes/dataset_analysis.md'ye markdown olarak kaydedilir
"""

from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────
SUBPIPE_ROOT = Path("/home/alp/thesis/datasets/SubPipe")
OUTPUT_MD    = Path("/home/alp/thesis/notes/dataset_analysis.md")

CHUNKS       = [f"Chunk{i}" for i in range(5)]
MODALITIES   = ["LF", "HF"]
MODALITY_DIR = {"LF": "SSS_LF_images", "HF": "SSS_HF_images"}
# ───────────────────────────────────────────────────────────────────────────


def analyze_chunk(chunk: str, modality: str) -> dict:
    chunk_dir = SUBPIPE_ROOT / chunk / MODALITY_DIR[modality]
    img_dir   = chunk_dir / "Image"
    lbl_dir   = chunk_dir / "YOLO_Annotation"

    stats = {
        "image_count": 0, "annotation_count": 0, "annotated_image_count": 0,
        "img_width": None, "img_height": None,
        "cx_mean": None, "cx_std": None, "cy_mean": None, "cy_std": None,
        "w_mean": None, "w_std": None, "h_mean": None, "h_std": None,
    }

    if not img_dir.exists():
        return stats

    images = sorted(img_dir.glob("*.pbm"))
    stats["image_count"] = len(images)

    if images:
        first = Image.open(images[0])
        stats["img_width"], stats["img_height"] = first.size

    cx, cy, w, h = [], [], [], []
    annotated = 0

    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue

        lines = lbl.read_text().strip().split("\n")
        had_ann = False
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cx.append(float(parts[1]))
                cy.append(float(parts[2]))
                w.append(float(parts[3]))
                h.append(float(parts[4]))
                had_ann = True
        if had_ann:
            annotated += 1

    stats["annotation_count"]       = len(cx)
    stats["annotated_image_count"]  = annotated

    if cx:
        stats["cx_mean"], stats["cx_std"] = float(np.mean(cx)), float(np.std(cx))
        stats["cy_mean"], stats["cy_std"] = float(np.mean(cy)), float(np.std(cy))
        stats["w_mean"],  stats["w_std"]  = float(np.mean(w)),  float(np.std(w))
        stats["h_mean"],  stats["h_std"]  = float(np.mean(h)),  float(np.std(h))

    return stats


def fmt(val, spec=".3f") -> str:
    return "N/A" if val is None else f"{val:{spec}}"


def build_report() -> str:
    data = {m: {c: analyze_chunk(c, m) for c in CHUNKS} for m in MODALITIES}

    lines: list[str] = []
    lines.append("# SubPipe Dataset Analizi")
    lines.append(f"\n_Oluşturulma: {datetime.now():%Y-%m-%d %H:%M}_")
    lines.append(f"\nKaynak: `{SUBPIPE_ROOT}`")

    for mod in MODALITIES:
        full_name = "Düşük Frekans (455 kHz)" if mod == "LF" else "Yüksek Frekans (900 kHz)"
        lines.append(f"\n## {mod} — Side-Scan Sonar, {full_name}")

        sample = next((c for c in CHUNKS if data[mod][c]["img_width"]), None)
        if sample:
            w = data[mod][sample]["img_width"]
            h = data[mod][sample]["img_height"]
            lines.append(f"\n**Görüntü boyutu:** {w} × {h} piksel")

        # Tablo 1: chunk genel sayıları
        lines.append("\n### Chunk başına özet\n")
        lines.append("| Chunk | Görüntü | Anotasyon | Anotasyonlu görüntü | Anot. yoğunluğu |")
        lines.append("|-------|---------|-----------|---------------------|------------------|")

        tot_img = tot_ann = tot_lbl = 0
        for c in CHUNKS:
            s = data[mod][c]
            ratio = (s["annotated_image_count"] / s["image_count"] * 100) if s["image_count"] else 0
            lines.append(
                f"| {c} | {s['image_count']} | {s['annotation_count']} | "
                f"{s['annotated_image_count']} | {ratio:.1f}% |"
            )
            tot_img += s["image_count"]
            tot_ann += s["annotation_count"]
            tot_lbl += s["annotated_image_count"]
        ratio_total = (tot_lbl / tot_img * 100) if tot_img else 0
        lines.append(
            f"| **TOPLAM** | **{tot_img}** | **{tot_ann}** | **{tot_lbl}** | "
            f"**{ratio_total:.1f}%** |"
        )

        # Tablo 2: pozisyon
        lines.append("\n### Boru pozisyonu (normalize koordinat)\n")
        lines.append("| Chunk | cx ort. | cx std | cy ort. | cy std |")
        lines.append("|-------|---------|--------|---------|--------|")
        for c in CHUNKS:
            s = data[mod][c]
            lines.append(
                f"| {c} | {fmt(s['cx_mean'])} | {fmt(s['cx_std'])} | "
                f"{fmt(s['cy_mean'])} | {fmt(s['cy_std'])} |"
            )

        # Tablo 3: kutu boyutları
        lines.append("\n### Kutu boyutları (normalize)\n")
        lines.append("| Chunk | w ort. | w std | h ort. | h std |")
        lines.append("|-------|--------|-------|--------|-------|")
        for c in CHUNKS:
            s = data[mod][c]
            lines.append(
                f"| {c} | {fmt(s['w_mean'])} | {fmt(s['w_std'])} | "
                f"{fmt(s['h_mean'])} | {fmt(s['h_std'])} |"
            )

    lines.append("\n---")
    lines.append("\n_Notlar:_")
    lines.append("- _cx, cy: kutu merkezi normalize koordinat (0=sol/üst, 1=sağ/alt)._")
    lines.append("- _w, h: kutu genişliği ve yüksekliği görüntü boyutuna göre normalize._")
    lines.append("- _Düşük std: chunk içinde boru pozisyonu/boyutu kararlı._")
    lines.append("- _h değeri 1'e yakınsa boru görüntünün dikey ekseninin tamamını kaplıyor._")

    return "\n".join(lines)


def main() -> None:
    report = build_report()
    print(report)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(report)
    print(f"\n[Kaydedildi: {OUTPUT_MD}]")


if __name__ == "__main__":
    main()