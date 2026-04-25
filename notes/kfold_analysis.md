# SubPipe K-Fold Cross Validation Sonuçları

_Oluşturulma: 2026-04-26 00:23_

Kaynak: `/home/alp/thesis/results/kfold_results.csv`

**Tamamlanan eğitim:** 10  |  **Fold'lar:** [0, 1, 2, 3, 4]  |  **Modaliteler:** ['HF', 'LF']

## 1. Fold-by-Fold Sonuçlar


### LF

| Fold | Test Chunk | Train img | Test img | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 | test P | test R | best epoch | süre (dk) |
|------|------------|-----------|----------|-----------|---------------|------------|----------------|--------|--------|-------------|------------|
| 0 | Chunk0 | 3451 | 1055 | 0.9884 | 0.7473 | 0.9284 | 0.5651 | 0.983 | 0.881 | 27 | 14.5 |
| 1 | Chunk1 | 3914 | 541 | 0.9934 | 0.7825 | 0.8844 | 0.5637 | 1.000 | 0.864 | 36 | 16.8 |
| 2 | Chunk2 | 3916 | 539 | 0.9911 | 0.7696 | 0.8305 | 0.5020 | 0.976 | 0.771 | 24 | 16.8 |
| 3 | Chunk3 | 4231 | 189 | 0.9895 | 0.7616 | 0.6775 | 0.1078 | 0.699 | 0.742 | 44 | 18.2 |
| 4 | Chunk4 | 2092 | 2566 | 0.9948 | 0.7968 | 0.9090 | 0.4396 | 0.937 | 0.870 | 46 | 9.1 |

### HF

| Fold | Test Chunk | Train img | Test img | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 | test P | test R | best epoch | süre (dk) |
|------|------------|-----------|----------|-----------|---------------|------------|----------------|--------|--------|-------------|------------|
| 0 | Chunk0 | 3617 | 903 | 0.9924 | 0.7177 | 0.9225 | 0.4873 | 0.946 | 0.880 | 44 | 17.4 |
| 1 | Chunk1 | 3943 | 541 | 0.9936 | 0.7332 | 0.8540 | 0.2573 | 0.980 | 0.814 | 41 | 19.3 |
| 2 | Chunk2 | 3945 | 539 | 0.9943 | 0.7490 | 0.8832 | 0.4515 | 0.979 | 0.757 | 41 | 19.6 |
| 3 | Chunk3 | 4259 | 190 | 0.9906 | 0.7415 | 0.9673 | 0.3316 | 0.956 | 0.931 | 30 | 21.5 |
| 4 | Chunk4 | 1956 | 2749 | 0.9844 | 0.7516 | 0.9005 | 0.4253 | 0.923 | 0.847 | 45 | 10.0 |

## 2. Modalite Başına Özet (mean ± std)

| Modalite | n | Test mAP50 | Test mAP50-95 | Test Precision | Test Recall | Val mAP50 | Val mAP50-95 |
|----------|---|----------------|----------------|----------------|----------------|----------------|----------------|
| **LF** | 5 | 0.8460 ± 0.1011 | 0.4356 ± 0.1904 | 0.9188 ± 0.1252 | 0.8258 ± 0.0643 | 0.9914 ± 0.0027 | 0.7716 ± 0.0190 |
| **HF** | 5 | 0.9055 ± 0.0427 | 0.3906 ± 0.0942 | 0.9567 ± 0.0239 | 0.8458 ± 0.0659 | 0.9911 ± 0.0040 | 0.7386 ± 0.0137 |

## 3. LF vs HF Karşılaştırması (Test seti üzerinden)

| Metrik | LF | HF | Δ (HF − LF) |
|--------|------|------|-------------|
| Test mAP50 | 0.8460 | 0.9055 | +0.0595 |
| Test mAP50-95 | 0.4356 | 0.3906 | -0.0450 |
| Test Precision | 0.9188 | 0.9567 | +0.0379 |
| Test Recall | 0.8258 | 0.8458 | +0.0200 |

## 4. Chunk3 Anomalisi (her fold ayrı analiz)

_Dataset analizinde Chunk3'te boru kutu yüksekliğinin diğer chunk'ların yarısı olduğu (h≈0.45 vs ~0.91) tespit edildi. Aşağıda Chunk3 test seti olduğunda sonuçlar var._

| Modalite | test mAP50 | test mAP50-95 | P | R |
|----------|------------|---------------|------|------|
| HF | 0.9673 | 0.3316 | 0.956 | 0.931 |
| LF | 0.6775 | 0.1078 | 0.699 | 0.742 |

---

_Not: Chunk3 test fold'unda anotasyon sayısı çok düşük (~30) olduğundan rakamlar diğer fold'lara kıyasla daha gürültülüdür. Standart sapma değerleri yorumlanırken bu göz önünde bulundurulmalıdır._