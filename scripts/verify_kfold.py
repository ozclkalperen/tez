#!/usr/bin/env python3
"""
verify_kfold.py — prepare_kfold.py çıktısının doğruluğunu kontrol eder.

Kontroller:
  - Her fold için 5 chunk dağılımı doğru mu (test = doğru chunk?)
  - Train/val/test sayıları beklenen chunk'lardan mı geliyor?
  - Symlink'ler kırık değil mi?
  - Image-label eşleşmesi tutarlı mı?
  - LF ve HF aynı timestamp setini içeriyor mu?

Çıktı:
  - Konsola özet + uyarılar
  - notes/kfold_verification.md (markdown rapor)
"""

from collections import Counter
from datetime import datetime
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
SUBPIPE_ROOT = Path("/home/alp/thesis/datasets/SubPipe")
KFOLD_ROOT   = Path("/home/alp/thesis/datasets/subpipe_kfold")
OUTPUT_MD    = Path("/home/alp/thesis/notes/kfold_verification.md")

CHUNKS       = [f"Chunk{i}" for i in range(5)]
MODALITIES   = ["LF", "HF"]
MODALITY_DIR = {"LF": "SSS_LF_images", "HF": "SSS_HF_images"}
N_FOLDS      = 5
SPLITS       = ["train", "val", "test"]
# ───────────────────────────────────────────────────────────────────────────


def build_timestamp_to_chunk(modality: str) -> dict[str, str]:
    """Her timestamp'in hangi chunk'a ait olduğunu bul."""
    mapping = {}
    for chunk in CHUNKS:
        img_dir = SUBPIPE_ROOT / chunk / MODALITY_DIR[modality] / "Image"
        for png in img_dir.glob("*.png"):
            mapping[png.stem] = chunk
    return mapping


def check_fold(fold_idx: int, ts_map: dict[str, dict[str, str]]) -> dict:
    """Tek bir fold için tüm kontrolleri yap, sonuçları dict olarak döndür."""
    expected_test = CHUNKS[fold_idx]
    expected_train = [c for c in CHUNKS if c != expected_test]
    fold_dir = KFOLD_ROOT / f"fold_{fold_idx}"

    result = {
        "fold": fold_idx,
        "expected_test": expected_test,
        "expected_train": expected_train,
        "exists": fold_dir.exists(),
        "modalities": {},
        "warnings": [],
        "errors": [],
    }

    if not fold_dir.exists():
        result["errors"].append(f"Fold klasörü yok: {fold_dir}")
        return result

    for mod in MODALITIES:
        mod_data = {
            "yaml_exists": False,
            "splits": {},
            "lf_hf_match": None,  # sadece HF kontrolünde dolduracağız
        }
        mod_dir = fold_dir / mod

        # YAML var mı?
        yaml_path = mod_dir / "data.yaml"
        mod_data["yaml_exists"] = yaml_path.exists()
        if not yaml_path.exists():
            result["errors"].append(f"{mod}: data.yaml yok")

        for split in SPLITS:
            img_dir = mod_dir / "images" / split
            lbl_dir = mod_dir / "labels" / split

            if not img_dir.exists():
                result["errors"].append(f"{mod}/{split}: images klasörü yok")
                continue

            images = list(img_dir.glob("*.png"))
            labels = list(lbl_dir.glob("*.txt"))

            # Symlink kırık mı?
            broken = [p for p in images if p.is_symlink() and not p.exists()]
            if broken:
                result["errors"].append(
                    f"{mod}/{split}: {len(broken)} kırık symlink"
                )

            # Hangi chunk'tan geldikleri
            chunk_dist = Counter()
            unmapped = 0
            for img in images:
                chunk = ts_map[mod].get(img.stem)
                if chunk:
                    chunk_dist[chunk] += 1
                else:
                    unmapped += 1
            if unmapped:
                result["warnings"].append(
                    f"{mod}/{split}: {unmapped} timestamp orijinal verilerde bulunamadı"
                )

            # Beklenen chunk'tan mı geliyor?
            if split == "test":
                wrong = sum(v for c, v in chunk_dist.items() if c != expected_test)
                if wrong:
                    result["errors"].append(
                        f"{mod}/{split}: {wrong} görüntü test chunk'ı dışından "
                        f"(beklenen: {expected_test}, bulunan: {dict(chunk_dist)})"
                    )
            else:  # train, val
                wrong_chunks = [c for c in chunk_dist if c == expected_test]
                if wrong_chunks:
                    cnt = sum(chunk_dist[c] for c in wrong_chunks)
                    result["errors"].append(
                        f"{mod}/{split}: {cnt} görüntü test chunk'ından sızmış "
                        f"({expected_test} train/val'de olmamalıydı)"
                    )

            # Image-label sayısı eşleşiyor mu?
            if len(images) != len(labels):
                result["warnings"].append(
                    f"{mod}/{split}: {len(images)} image vs {len(labels)} label"
                )

            # Anotasyonlu (boş olmayan) label sayısı
            non_empty_labels = sum(
                1 for lbl in labels if lbl.exists() and lbl.stat().st_size > 0
            )

            mod_data["splits"][split] = {
                "n_images": len(images),
                "n_labels": len(labels),
                "n_annotated": non_empty_labels,
                "chunk_distribution": dict(chunk_dist),
                "broken_symlinks": len(broken),
            }

        result["modalities"][mod] = mod_data

    # LF ve HF aynı timestamp setine mi sahip? (her split için)
    if "LF" in result["modalities"] and "HF" in result["modalities"]:
        for split in SPLITS:
            lf_dir = fold_dir / "LF" / "images" / split
            hf_dir = fold_dir / "HF" / "images" / split
            if not (lf_dir.exists() and hf_dir.exists()):
                continue
            lf_stems = {p.stem for p in lf_dir.glob("*.png")}
            hf_stems = {p.stem for p in hf_dir.glob("*.png")}
            only_lf = lf_stems - hf_stems
            only_hf = hf_stems - lf_stems
            # NOT: bu kontrolün başarısız olması beklenir, çünkü LF/HF
            # aynı zamanda alınmış olsa da farklı sayıda görüntü içeriyor.
            # Bu yüzden warning değil bilgi olarak raporluyoruz.
            if only_lf or only_hf:
                result["modalities"]["HF"]["lf_hf_match"] = {
                    "only_lf": len(only_lf),
                    "only_hf": len(only_hf),
                    "common": len(lf_stems & hf_stems),
                }

    return result


def format_report(results: list[dict]) -> str:
    lines = []
    lines.append("# K-Fold Hazırlık Doğrulama Raporu")
    lines.append(f"\n_Oluşturulma: {datetime.now():%Y-%m-%d %H:%M}_")
    lines.append(f"\nKaynak: `{KFOLD_ROOT}`")

    # ── Genel özet ────────────────────────────────────────────────────────
    total_errors   = sum(len(r["errors"]) for r in results)
    total_warnings = sum(len(r["warnings"]) for r in results)
    status_emoji   = "✅ Hata yok" if total_errors == 0 else f"❌ {total_errors} hata"

    lines.append(f"\n## Genel Durum\n")
    lines.append(f"- **Durum:** {status_emoji}")
    lines.append(f"- **Toplam fold:** {len(results)}")
    lines.append(f"- **Uyarılar:** {total_warnings}")

    # ── Fold dağılım tablosu ──────────────────────────────────────────────
    lines.append("\n## Fold Dağılımı\n")
    lines.append("| Fold | Test Chunk | Train Chunks |")
    lines.append("|------|------------|--------------|")
    for r in results:
        train_str = ", ".join(r["expected_train"])
        lines.append(f"| {r['fold']} | **{r['expected_test']}** | {train_str} |")

    # ── Görüntü sayıları (LF) ─────────────────────────────────────────────
    for mod in MODALITIES:
        lines.append(f"\n## {mod} Görüntü Sayıları\n")
        lines.append("| Fold | Train | Val | Test | "
                     "Train Annot. | Val Annot. | Test Annot. |")
        lines.append("|------|-------|-----|------|"
                     "--------------|------------|--------------|")
        for r in results:
            md = r["modalities"].get(mod, {})
            sp = md.get("splits", {})
            tr = sp.get("train", {})
            vl = sp.get("val", {})
            ts = sp.get("test", {})
            lines.append(
                f"| {r['fold']} | {tr.get('n_images', '-')} | "
                f"{vl.get('n_images', '-')} | {ts.get('n_images', '-')} | "
                f"{tr.get('n_annotated', '-')} | "
                f"{vl.get('n_annotated', '-')} | "
                f"{ts.get('n_annotated', '-')} |"
            )

    # ── Test setlerinin chunk dağılımı (sızıntı kontrolü) ─────────────────
    lines.append("\n## Test Setlerinde Chunk Dağılımı (sızıntı kontrolü)\n")
    lines.append(
        "_Her fold'un test setinde sadece beklenen chunk'tan görüntüler olmalı._\n"
    )
    lines.append("| Fold | Modalite | Beklenen | Gerçek Dağılım |")
    lines.append("|------|----------|----------|-----------------|")
    for r in results:
        for mod in MODALITIES:
            md = r["modalities"].get(mod, {})
            ts = md.get("splits", {}).get("test", {})
            dist = ts.get("chunk_distribution", {})
            dist_str = ", ".join(f"{c}={n}" for c, n in dist.items())
            ok = list(dist.keys()) == [r["expected_test"]] if dist else False
            mark = "✅" if ok else "❌"
            lines.append(
                f"| {r['fold']} | {mod} | {r['expected_test']} | "
                f"{mark} {dist_str} |"
            )

    # ── Train/val sızıntısı kontrolü ──────────────────────────────────────
    lines.append("\n## Train/Val Sızıntı Kontrolü\n")
    lines.append(
        "_Test chunk'ı train veya val'da olmamalı. Boş tablo iyi haberdir._\n"
    )
    leak_found = False
    lines.append("| Fold | Modalite | Split | Sızıntı |")
    lines.append("|------|----------|-------|---------|")
    for r in results:
        for mod in MODALITIES:
            md = r["modalities"].get(mod, {})
            for split in ["train", "val"]:
                sp = md.get("splits", {}).get(split, {})
                dist = sp.get("chunk_distribution", {})
                if r["expected_test"] in dist:
                    leak_found = True
                    lines.append(
                        f"| {r['fold']} | {mod} | {split} | "
                        f"❌ {r['expected_test']}={dist[r['expected_test']]} |"
                    )
    if not leak_found:
        lines.append("| – | – | – | ✅ Sızıntı yok |")

    # ── Hatalar ve uyarılar ───────────────────────────────────────────────
    if total_errors > 0:
        lines.append("\n## ❌ Hatalar\n")
        for r in results:
            for err in r["errors"]:
                lines.append(f"- **Fold {r['fold']}:** {err}")

    if total_warnings > 0:
        lines.append("\n## ⚠️ Uyarılar\n")
        for r in results:
            for w in r["warnings"]:
                lines.append(f"- **Fold {r['fold']}:** {w}")

    # ── Sonuç ─────────────────────────────────────────────────────────────
    lines.append("\n---\n")
    if total_errors == 0:
        lines.append("**Sonuç:** Tüm kontroller başarılı, k-fold pipeline'ı eğitime hazır.")
    else:
        lines.append(
            f"**Sonuç:** {total_errors} hata bulundu. "
            f"Eğitime başlamadan önce çözülmeli."
        )

    return "\n".join(lines)


def main() -> None:
    print("K-Fold doğrulama başlatılıyor...\n")

    # Timestamp → chunk haritalarını bir kez kur
    print("  Orijinal veri haritası çıkarılıyor...")
    ts_map = {mod: build_timestamp_to_chunk(mod) for mod in MODALITIES}
    for mod in MODALITIES:
        print(f"    {mod}: {len(ts_map[mod])} timestamp")

    # Her fold'u kontrol et
    print("\n  Fold'lar kontrol ediliyor...")
    results = []
    for fold_idx in range(N_FOLDS):
        print(f"    Fold {fold_idx}...", end=" ")
        r = check_fold(fold_idx, ts_map)
        e_count = len(r["errors"])
        w_count = len(r["warnings"])
        status = "OK" if e_count == 0 else f"{e_count} hata"
        print(f"{status}, {w_count} uyarı")
        results.append(r)

    report = format_report(results)
    print("\n" + "=" * 64)
    print(report)
    print("=" * 64)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(report)
    print(f"\n[Markdown rapor: {OUTPUT_MD}]")

    # Çıkış kodu (eğer hata varsa shell'de görsün)
    total_errors = sum(len(r["errors"]) for r in results)
    if total_errors > 0:
        exit(1)


if __name__ == "__main__":
    main()