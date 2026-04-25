# Master Tezi Proje Bağlamı

## Proje Özeti
**Konu:** Spiking Neural Network (SNN) tabanlı sualtı sonar görüntülerinde
boru tespiti.

**Veri seti:** SubPipe (REMARO/OceanScan-MST). Side-scan sonar görüntüleri.
Tek sınıf object detection: pipe.

**Mimari:** YOLO26 (Ultralytics, Ocak 2026'da çıktı). Önce ANN baseline
kurulacak, sonra ANN-to-SNN dönüşümü ile SNN versiyonu test edilecek.
SpikingJelly kütüphanesi kullanılacak.

**Asıl tez sorusu:** "Chunk-bazlı domain shift altında SNN, ANN'e göre
daha iyi generalize edebilir mi?"

## Veri Seti Detayları

### Yapı
- 5 chunk, her biri AUV'nin boru üzerinde bir geçişi (~7-9 dk)
- LF (Low Frequency, 455 kHz): 2500x500 px, .pbm formatında
- HF (High Frequency, 900 kHz): 5000x500 px, .pbm formatında
- Anotasyonlar zaten YOLO formatında verili
- Toplam: ~5000 görüntü/frekans, ~3100 anotasyon/frekans

### Konum: `/home/alp/thesis/datasets/SubPipe/`
Her chunk klasörü içinde `SSS_LF_images/` ve `SSS_HF_images/` var.

### Chunk Karakteristikleri (analiz tamamlandı)
| Chunk | Anot. yoğ. | cx (HF) | h (HF) | Not |
|-------|-----------|---------|--------|-----|
| Chunk0 | %63 | 0.745 (sağ) | 0.91 | Yoğun, sağ pozisyon |
| Chunk1 | %46 | 0.404 (sol-orta) | 0.91 | Orta yoğun |
| Chunk2 | %81 | 0.438 (sol-orta) | 0.95 | Çok yoğun |
| Chunk3 | %15 | 0.221 (sol uç) | **0.45** | Anomali — kutu yükseliği yarı yarıya |
| Chunk4 | %65 | 0.360 (sol-değişken) | 0.93 | En büyük chunk |

**Chunk3 anomalisi:** Diğer chunk'larda boru görüntünün dikey ~%90'ını
kaplarken, Chunk3'te sadece ~%45'ini kaplıyor. Olası sebep AUV
yüksekliği veya görüş açısı farkı. Bu, Chunk3'ün generalization
testinde neden zor olduğunu açıklar.

## Şimdiye Kadar Yapılanlar

### Eski deneyler (sonuçlar elde edildi)
1. Pilot eğitim: Chunk0 LF random split → mAP50=0.99 (data leakage,
   ardışık kareler val'a düşüyor)
2. HF baseline (Chunk0+4 train, Chunk1 val, Chunk2 test):
   val mAP50=0.95, mAP50-95=0.37, robust (Chunk3) mAP50=0.94
3. LF baseline (aynı split): val mAP50=0.92, mAP50-95=0.59,
   robust mAP50=0.26 (LF generalization'da çöküyor!)
4. HF 1280 (imgsz arttırıldı): val mAP50-95=0.54 (büyük iyileşme)
5. LF 1280: robust mAP50=0.55 (640'taki 0.26'dan iki kat iyi)

**Anahtar bulgu:** İmgsz=640 stretched resize paper'ın kullandığı yöntem
ama bu LF için generalization felaketi yaratıyor. İmgsz=1280 trade-off'lu
iyileşme sağlıyor.

### Verilen kararlar (yeni plan)
- **5-fold cross validation** yapacağız (Chunk3 dahil, fold-by-fold raporla)
- **Early stopping kapalı**, sabit 50 epoch
- **İç val:** Train chunk'larından random %10 (best epoch seçimi için)
- **imgsz=640**, batch=16, workers=8
- **Toplam eğitim:** 5 fold × 2 modalite (LF, HF) = 10 eğitim, ~1.5 saat

### Klasör yapısı
/home/alp/thesis/
├── datasets/
│   ├── SubPipe/                    # Orijinal kaynak — DOKUNMA
│   └── subpipe_kfold/              # K-fold için hazırlanacak
├── scripts/
│   └── dataset_analysis.py         # Tamamlandı, çalışıyor
├── runs/                           # YOLO eğitim çıktıları
├── results/                        # CSV ve özet tablolar
└── notes/
├── PROJECT_CONTEXT.md          # Bu dosya
└── dataset_analysis.md         # Otomatik üretildi

## Sırada Ne Var

### Yapılacak script'ler
1. `prepare_kfold.py` — 5 fold için klasör/YAML hazırlığı
2. `run_kfold.py` — 10 eğitimi sırayla yapar, sonuçları CSV'ye yazar
3. `summarize_results.py` — CSV'den mean ± std tablosu üretir

### Daha sonra (SNN aşaması)
4. ANN-to-SNN dönüşümü (SpikingJelly converter)
5. SNN eğitimi (timestep ablation: T=8, 16, 32, 64, 128)
6. Enerji/parametre karşılaştırma analizi

### Opsiyonel (zaman varsa)
- SAHI tiling deneyi (orijinal çözünürlüğü korumak için)
- Multimodal LF+HF füzyon

## Önemli Teknik Notlar
- GPU: RTX 4070, 12 GB VRAM (rahat çalışıyor batch=16, imgsz=640)
- Ultralytics 8.4.41 yüklü, YOLO26 destekli
- PyTorch 2.5.1 + CUDA 12.1
- Python 3.10
- SpikingJelly henüz kurulu değil, SNN aşamasında lazım olacak

## Tez Yapısı (Taslak)
1. Giriş & literatür (sualtı sonar, SNN, generalization problemi)
2. Veri seti analizi (chunk istatistikleri)
3. Metodoloji (5-fold CV protokolü, model seçimi)
4. ANN baseline sonuçları
5. SNN dönüşümü ve sonuçları
6. Karşılaştırmalı analiz (mAP, enerji, parametre)
7. Tartışma & sonuç
