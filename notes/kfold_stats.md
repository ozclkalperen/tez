# 5-Fold Cross Validation — Bölünme İstatistikleri

`prepare_kfold.py` çalıştırıldı: 2026-04-25 · Seed=42 · Val=%10 (train havuzundan random)

---

## Fold Atamaları

| Fold | Test Chunk | Train Havuzu |
|------|-----------|--------------|
| 0 | Chunk0 | Chunk1, Chunk2, Chunk3, Chunk4 |
| 1 | Chunk1 | Chunk0, Chunk2, Chunk3, Chunk4 |
| 2 | Chunk2 | Chunk0, Chunk1, Chunk3, Chunk4 |
| 3 | Chunk3 | Chunk0, Chunk1, Chunk2, Chunk4 |
| 4 | Chunk4 | Chunk0, Chunk1, Chunk2, Chunk3 |

---

## LF (Low Frequency, 455 kHz)

| Fold | Test | Train img | Train lbl (%) | Val img | Val lbl (%) | Test img | Test lbl (%) |
|------|------|-----------|--------------|---------|------------|----------|-------------|
| 0 | Chunk0 | 3451 | 2121 (61%) | 384 | 243 (63%) | 1055 | 696 (66%) |
| 1 | Chunk1 | 3914 | 2540 (65%) | 435 | 277 (64%) | 541 | 243 (45%) |
| 2 | Chunk2 | 3916 | 2351 (60%) | 435 | 254 (58%) | 539 | 455 (84%) |
| **3** | **Chunk3** | 4231 | 2737 (65%) | 470 | 292 (62%) | **189** | **31 (16%)** |
| 4 | Chunk4 | 2092 | 1270 (61%) | 232 | 155 (67%) | 2566 | 1635 (64%) |

## HF (High Frequency, 900 kHz)

| Fold | Test | Train img | Train lbl (%) | Val img | Val lbl (%) | Test img | Test lbl (%) |
|------|------|-----------|--------------|---------|------------|----------|-------------|
| 0 | Chunk0 | 3617 | 2243 (62%) | 402 | 253 (63%) | 903 | 566 (63%) |
| 1 | Chunk1 | 3943 | 2542 (64%) | 438 | 273 (62%) | 541 | 247 (46%) |
| 2 | Chunk2 | 3945 | 2367 (60%) | 438 | 260 (59%) | 539 | 435 (81%) |
| **3** | **Chunk3** | 4259 | 2723 (64%) | 473 | 310 (66%) | **190** | **29 (15%)** |
| 4 | Chunk4 | 1956 | 1165 (60%) | 217 | 112 (52%) | 2749 | 1785 (65%) |

---

## Notlar

**Fold 3 (Chunk3 test) — zor fold:**
- Test setinde yalnızca ~%15-16 görüntü anotasyonlu (diğer foldlar %45-84 arası)
- Chunk3 bbox yüksekliği HF için h≈0.45 (diğerleri h≈0.91) — AUV yüksekliği/açı anomalisi
- Bu fold SNN'nin ve ANN'in Chunk3'e generalize edip edemediğini ölçecek

**Fold 4 (Chunk4 test) — en büyük test seti:**
- LF test: 2566 img, HF test: 2749 img (toplam görüntünün ~%55'i)
- Train havuzu küçülüyor (sadece Chunk0-3): train=~2000 img, öğrenme kapasitesi düşebilir

**Veri sızıntısı yok:**
- Her split tamamen chunk bazında ayrılmış (ardışık kare leak riski yok)
- Val seti train chunk havuzundan random seçildi, test chunk'ından bağımsız
