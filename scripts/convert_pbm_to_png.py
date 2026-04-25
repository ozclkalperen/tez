#!/usr/bin/env python3
"""
convert_pbm_to_png.py — SubPipe içindeki tüm .pbm dosyalarını .png'ye çevirir.

Her .pbm dosyasının yanına aynı isimde .png yazılır. Mevcut .png varsa atlar.
Tek seferlik, dataset hazırlığından önce çalıştırılır.
"""

from pathlib import Path
from PIL import Image

BASE_DIR = Path("/home/alp/thesis/datasets/SubPipe")


def main() -> None:
    print(f"Kaynak: {BASE_DIR}")
    print("PBM → PNG dönüşümü başlıyor...\n")

    pbm_files = list(BASE_DIR.rglob("*.pbm"))
    total = len(pbm_files)
    converted = 0
    skipped = 0

    for i, pbm in enumerate(pbm_files, 1):
        png = pbm.with_suffix(".png")

        if png.exists():
            skipped += 1
        else:
            img = Image.open(pbm)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(png)
            converted += 1

        if i % 500 == 0 or i == total:
            print(f"  [{i:>5}/{total}]  dönüştürüldü: {converted}  atlandı: {skipped}")

    print(f"\nTamamlandı.")
    print(f"  Toplam PBM   : {total}")
    print(f"  Yeni PNG     : {converted}")
    print(f"  Zaten varolan: {skipped}")


if __name__ == "__main__":
    main()