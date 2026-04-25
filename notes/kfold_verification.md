# K-Fold Hazırlık Doğrulama Raporu

_Oluşturulma: 2026-04-25 20:50_

Kaynak: `/home/alp/thesis/datasets/subpipe_kfold`

## Genel Durum

- **Durum:** ✅ Hata yok
- **Toplam fold:** 5
- **Uyarılar:** 0

## Fold Dağılımı

| Fold | Test Chunk | Train Chunks |
|------|------------|--------------|
| 0 | **Chunk0** | Chunk1, Chunk2, Chunk3, Chunk4 |
| 1 | **Chunk1** | Chunk0, Chunk2, Chunk3, Chunk4 |
| 2 | **Chunk2** | Chunk0, Chunk1, Chunk3, Chunk4 |
| 3 | **Chunk3** | Chunk0, Chunk1, Chunk2, Chunk4 |
| 4 | **Chunk4** | Chunk0, Chunk1, Chunk2, Chunk3 |

## LF Görüntü Sayıları

| Fold | Train | Val | Test | Train Annot. | Val Annot. | Test Annot. |
|------|-------|-----|------|--------------|------------|--------------|
| 0 | 3451 | 384 | 1055 | 2121 | 243 | 696 |
| 1 | 3914 | 435 | 541 | 2540 | 277 | 243 |
| 2 | 3916 | 435 | 539 | 2351 | 254 | 455 |
| 3 | 4231 | 470 | 189 | 2737 | 292 | 31 |
| 4 | 2092 | 232 | 2566 | 1270 | 155 | 1635 |

## HF Görüntü Sayıları

| Fold | Train | Val | Test | Train Annot. | Val Annot. | Test Annot. |
|------|-------|-----|------|--------------|------------|--------------|
| 0 | 3617 | 402 | 903 | 2243 | 253 | 566 |
| 1 | 3943 | 438 | 541 | 2542 | 273 | 247 |
| 2 | 3945 | 438 | 539 | 2367 | 260 | 435 |
| 3 | 4259 | 473 | 190 | 2723 | 310 | 29 |
| 4 | 1956 | 217 | 2749 | 1165 | 112 | 1785 |

## Test Setlerinde Chunk Dağılımı (sızıntı kontrolü)

_Her fold'un test setinde sadece beklenen chunk'tan görüntüler olmalı._

| Fold | Modalite | Beklenen | Gerçek Dağılım |
|------|----------|----------|-----------------|
| 0 | LF | Chunk0 | ✅ Chunk0=1055 |
| 0 | HF | Chunk0 | ✅ Chunk0=903 |
| 1 | LF | Chunk1 | ✅ Chunk1=541 |
| 1 | HF | Chunk1 | ✅ Chunk1=541 |
| 2 | LF | Chunk2 | ✅ Chunk2=539 |
| 2 | HF | Chunk2 | ✅ Chunk2=539 |
| 3 | LF | Chunk3 | ✅ Chunk3=189 |
| 3 | HF | Chunk3 | ✅ Chunk3=190 |
| 4 | LF | Chunk4 | ✅ Chunk4=2566 |
| 4 | HF | Chunk4 | ✅ Chunk4=2749 |

## Train/Val Sızıntı Kontrolü

_Test chunk'ı train veya val'da olmamalı. Boş tablo iyi haberdir._

| Fold | Modalite | Split | Sızıntı |
|------|----------|-------|---------|
| – | – | – | ✅ Sızıntı yok |

---

**Sonuç:** Tüm kontroller başarılı, k-fold pipeline'ı eğitime hazır.