# SNN Probe Notu

Tarih: 2026-06-20

## Küçük Chunk2 Denemesi

Mevcut `datasets/SubPipe` klasörü tek chunk/flat yapı içeriyor. Bu veriyle
small-run dataset hazırlandı:

- Çıktı: `datasets/subpipe_single_chunk`
- LF split: train 377, val 81, test 81
- HF split: train 377, val 81, test 81

LF için 1 epoch CPU smoke eğitimi çalıştı:

- Checkpoint: `runs/single_chunk/chunk2_LF/weights/best.pt`
- Val mAP50: 0.5644
- Val mAP50-95: 0.2767
- Test mAP50: 0.6000
- Test mAP50-95: 0.3081

Bu değerler bilimsel sonuç değil; sadece veri/YOLO hattının çalıştığını
doğrulayan hızlı smoke testtir.

## SpikingJelly ANN-to-SNN Probe

Komut:

```bash
python3 scripts/snn_probe.py --mod LF --convert
```

Bulgular:

- Modelde 3 adet `SiLU` aktivasyonu var.
- SpikingJelly `ann2snn.Converter`, Ultralytics YOLO forward akışını
  `torch.fx.symbolic_trace` ile izlerken duruyor.
- Hata: `torch.fx.proxy.TraceError: Proxy object cannot be iterated`

Yorum:

Standart SpikingJelly ANN-to-SNN converter, Ultralytics YOLO26 modelini
doğrudan dönüştüremiyor. Bir sonraki adım, modeli tracing-friendly bir
alt grafa ayırmak veya YOLO bloklarını manuel spiking karşılıklarıyla
değiştiren özel bir dönüşüm hattı yazmak olmalı.

## Hybrid SNN Aktivasyon Denemesi

Standart converter başarısız olduğu için ara adım olarak `Conv.act` içindeki
`SiLU` aktivasyonları SpikingJelly `IFNode` ile değiştirildi ve YOLO validator
ile değerlendirildi:

```bash
python3 scripts/run_hybrid_snn.py \
  --scopes none,until0,all \
  --thresholds 0.1,0.5 \
  --timesteps 2 \
  --imgsz 320 \
  --batch 4 \
  --device cpu
```

Sonuç CSV'si: `results/single_chunk/hybrid_snn_sweep.csv`

| Scope | Threshold | T | Replaced | Test mAP50 | Test mAP50-95 |
|-------|-----------|---|----------|------------|---------------|
| none | 1.0 | 1 | 0 | 0.6000 | 0.3081 |
| until0 | 0.1 | 2 | 1 | 0.0000 | 0.0000 |
| until0 | 0.5 | 2 | 1 | 0.0000 | 0.0000 |
| all | 0.1 | 2 | 105 | 0.0000 | 0.0000 |
| all | 0.5 | 2 | 105 | 0.0000 | 0.0000 |

Yorum:

Sadece ilk aktivasyonu spike node'a çevirmek bile performansı sıfırlıyor.
Bu nedenle doğrudan `SiLU -> IFNode` değişimi, kalibrasyon/fine-tuning olmadan
uygun değil. Bir sonraki daha doğru yol:

1. ReLU/quantization-aware bir YOLO varyantı fine-tune etmek.
2. Aktivasyon istatistiği toplayıp threshold ve scale kalibrasyonu yapmak.
3. Backbone/neck'i spiking, detection head'i analog bırakan özel hibrit model
   üzerinde SNN-aware fine-tuning denemek.

## ReLU Checkpoint ve Hybrid SNN Sweep

Ultralytics trainer, `nc=1` override sırasında modeli YAML'dan yeniden kurduğu
için sadece bellekte yapılan `SiLU -> ReLU` değişimi korunmuyor. Bu nedenle
`scripts/run_single_chunk.py --activation relu` artık:

1. YOLO26n modelini yükler.
2. `Conv.act` içindeki 105 adet `SiLU` aktivasyonu `ReLU` ile değiştirir.
3. `model.yaml["activation"] = "torch.nn.ReLU(inplace=True)"` bilgisini yazar.
4. `runs/single_chunk/init/yolo26n_relu.pt` checkpoint'ini kaydeder.
5. Eğitimi bu checkpoint'ten başlatır.

Doğrulama:

- Checkpoint: `runs/single_chunk/chunk2_LF_relu/weights/best.pt`
- YAML activation: `torch.nn.ReLU(inplace=True)`
- Checkpoint içi `Conv.act`: 105 ReLU, 0 SiLU
- 1 epoch CPU ReLU smoke test: test mAP50 0.0356, test mAP50-95 0.0090

Bu düşük skor beklenen bir ara sonuçtur; SiLU ağırlıkları ReLU mimariye
aktarılıp yalnızca 1 epoch fine-tune edildi.

ReLU checkpoint üzerinde kısa hybrid SNN sweep:

```bash
python3 scripts/run_hybrid_snn.py \
  --checkpoint runs/single_chunk/chunk2_LF_relu/weights/best.pt \
  --scopes none,until0,until5,all \
  --source-activation ReLU \
  --thresholds 0.05,0.1,0.5 \
  --timesteps 2,4 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn_relu_sweep.csv
```

Özet:

| Scope | Best threshold | Best T | Replaced | Test mAP50 | Test mAP50-95 |
|-------|----------------|--------|----------|------------|---------------|
| none | 1.0 | 1 | 0 | 0.0356 | 0.0090 |
| until0 | 0.5 | 2/4 | 1 | 0.0001 | 0.0000 |
| until5 | 0.05/0.1/0.5 | 2/4 | 12 | 0.0000 | 0.0000 |
| all | 0.05/0.1/0.5 | 2/4 | 105 | 0.0000 | 0.0000 |

Yorum:

ReLU altyapısı artık doğru çalışıyor; `replaced` sayıları beklenen şekilde
1, 12 ve 105 geliyor. Fakat doğrudan `ReLU -> IFNode` değişimi, 1 epoch ReLU
model üzerinde pratik performans üretmiyor. Bir sonraki mantıklı adım daha
uzun ReLU fine-tuning ve sonrasında aktivasyon istatistiğine dayalı
threshold/scale kalibrasyonu eklemek.

## Uzun ReLU Fine-tuning Sonrası

20 epoch CPU ReLU fine-tune denemesi:

```bash
python3 scripts/run_single_chunk.py \
  --mod LF \
  --epochs 20 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --activation relu
```

Son checkpoint doğrulaması:

- Checkpoint: `runs/single_chunk/chunk2_LF_relu/weights/best.pt`
- YAML activation: `torch.nn.ReLU(inplace=True)`
- Checkpoint içi `Conv.act`: 105 ReLU, 0 SiLU

ANN sonuçları:

| Split | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| val | 0.9348 | 0.9054 | 0.9310 | 0.6687 |
| test | 0.8952 | 0.9491 | 0.9559 | 0.7389 |

Bu sonuç, ReLU mimarinin yeterli fine-tuning ile iyi çalıştığını gösteriyor.
Dolayısıyla önceki düşük 1 epoch ReLU sonucu sadece adaptasyonun eksik
olmasından kaynaklanıyordu.

Aynı iyi ReLU checkpoint üzerinde hybrid SNN sweep:

```bash
python3 scripts/run_hybrid_snn.py \
  --checkpoint runs/single_chunk/chunk2_LF_relu/weights/best.pt \
  --scopes none,until0,until5,all \
  --source-activation ReLU \
  --thresholds 0.05,0.1,0.5 \
  --timesteps 2,4 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn_relu_finetuned_sweep.csv
```

Özet:

| Scope | Thresholds | T | Replaced | Test mAP50 | Test mAP50-95 |
|-------|------------|---|----------|------------|---------------|
| none | 1.0 | 1 | 0 | 0.9559 | 0.7389 |
| until0 | 0.05/0.1/0.5 | 2/4 | 1 | 0.0000 | 0.0000 |
| until5 | 0.05/0.1/0.5 | 2/4 | 12 | 0.0000 | 0.0000 |
| all | 0.05/0.1/0.5 | 2/4 | 105 | 0.0000 | 0.0000 |

Yorum:

Uzun fine-tune, analog ReLU modelini güçlü hale getirdi; fakat direkt
`ReLU -> IFNode` değişimi hâlâ tamamen başarısız. Bu artık modelin zayıflığı
değil, dönüşümün ölçek/threshold uyumsuzluğu olduğunu gösteriyor. Bir sonraki
adım, IFNode öncesi aktivasyon istatistikleri toplayıp katman bazlı scale ve
threshold kalibrasyonu eklemek olmalı. Detection head'i analog bırakmak tek
başına yetmiyor; ilk backbone aktivasyonu bile sıfıra düşürüyor.

## Aktivasyon Kalibrasyonu ve İlk Pozitif Hybrid SNN Sonucu

Kalibrasyon eklenirken iki önemli uygulama detayı düzeltildi:

1. `HybridSNNModel.forward` içinde SpikingJelly state reset'i her timestep
   içinde değil, timestep döngüsünden önce bir kez yapılacak şekilde değişti.
   Böylece IFNode membran potansiyeli timestep'ler boyunca birikebiliyor.
2. Aktivasyon istatistikleri artık YOLO validator hook'ları üzerinden değil,
   `data.yaml` split klasöründen görüntüleri okuyup modele doğrudan forward
   vererek toplanıyor. Validator üzerinden alınan ilk ölçümde ortak aktivasyon
   nesnesi/hook yan etkileri ve sayısal uç değerler nedeniyle p99 değerleri
   güvenilir değildi. Manuel akışta 105 ReLU katmanı için p99 aralığı makul
   hale geldi: min 0.8883, ortalama 4.9503, max 60.9971.

Kalibrasyon yöntemi:

- Her `Conv.act` çıkışı için katman bazlı p99 aktivasyon değeri toplandı.
- IFNode eşiği `threshold_multiplier * layer_p99` olarak ayarlandı.
- Spike çıkışı, analog ReLU genliğini yaklaşık korumak için aynı eşik değeriyle
  ölçeklendi (`--spike-scale threshold`).

Kontrol komutu:

```bash
python3 scripts/run_hybrid_snn.py \
  --checkpoint runs/single_chunk/chunk2_LF_relu/weights/best.pt \
  --scopes none,until0,head \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --thresholds 0.25,0.5,1.0 \
  --timesteps 8 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn_relu_manual_calibrated_sweep.csv \
  --stats-out results/single_chunk/relu_activation_stats_manual.csv
```

Sonuç:

| Scope | Multiplier | Effective v_th | T | Replaced | Test mAP50 | Test mAP50-95 |
|-------|------------|----------------|---|----------|------------|---------------|
| none | 1.0 | 1.0000 | 1 | 0 | 0.9559 | 0.7389 |
| until0 | 0.25 | 4.3416 | 8 | 1 | 0.6099 | 0.3386 |
| until0 | 0.50 | 8.6831 | 8 | 1 | 0.5349 | 0.3675 |
| until0 | 1.00 | 17.3663 | 8 | 1 | 0.0013 | 0.0002 |
| head | 0.25/0.50/1.00 | katman bazlı | 8 | 36 | 0.0000 | 0.0000 |

Odaklanmış `until0` sweep:

```bash
python3 scripts/run_hybrid_snn.py \
  --checkpoint runs/single_chunk/chunk2_LF_relu/weights/best.pt \
  --scopes until0 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --thresholds 0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5 \
  --timesteps 8,16 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn_relu_until0_focused.csv \
  --stats-out results/single_chunk/relu_activation_stats_manual.csv
```

Özet:

| Multiplier | Effective v_th | T | Precision | Recall | Test mAP50 | Test mAP50-95 |
|------------|----------------|---|-----------|--------|------------|---------------|
| 0.20 | 3.4733 | 8 | 0.2475 | 0.3472 | 0.2915 | 0.1323 |
| 0.25 | 4.3416 | 8 | 0.2655 | 0.6528 | 0.6099 | 0.3386 |
| 0.30 | 5.2099 | 8 | 0.9750 | 0.5414 | 0.6361 | 0.4115 |
| 0.40 | 6.9465 | 8 | 0.9655 | 0.6528 | 0.7006 | 0.4936 |
| 0.40 | 6.9465 | 16 | 0.9919 | 0.6667 | 0.7420 | 0.5002 |
| 0.50 | 8.6831 | 16 | 0.8975 | 0.5556 | 0.6229 | 0.3886 |

Yorum:

Bu ilk pozitif hybrid SNN sonucudur. Önceki sıfır sonuçların ana nedeni
ReLU checkpoint'in zayıflığı değil, IFNode temporal reset ve aktivasyon
ölçek/eşik kalibrasyonunun eksik olmasıydı. Şimdilik yalnızca ilk backbone
aktivasyonu spiking yapıldığında analog ReLU modele göre hâlâ belirgin kayıp
var; fakat mAP50 0.7420 ve mAP50-95 0.5002 seviyesine ulaşmak, dönüşüm hattının
tamamen çalışmaz olmadığını gösteriyor. Detection head'i doğrudan spiking
yapmak ise bu ayarda hâlâ sıfır mAP üretiyor; sonraki adım, daha dar katman
grupları ve/veya SNN-aware fine-tuning ile kademeli dönüşüm denemek olmalı.

## Kademeli Blok Dönüşüm Haritası

Bir sonraki adım olarak ilk katmandan sonra top-level YOLO blokları tek tek ve
kümülatif aralıklar halinde spiking yapıldı. `scripts/run_hybrid_snn.py` içine
bu amaçla iki yeni scope eklendi:

- `onlyN`: yalnızca `model.N` altındaki aktivasyonları IFNode'a çevirir.
- `rangeA-B`: `model.A` ile `model.B` arasındaki top-level blokları çevirir.

Bu deneylerde sabit ayar olarak `p99` aktivasyon kalibrasyonu,
`--spike-scale threshold`, `threshold multiplier = 0.4` ve `T=16` kullanıldı.
Analog referans test sonucu yine mAP50 0.9559 ve mAP50-95 0.7389'dur.

Tek blok haritası:

```bash
python3 scripts/run_hybrid_snn.py \
  --checkpoint runs/single_chunk/chunk2_LF_relu/weights/best.pt \
  --scopes none,only1,only2,only3,only4,only5,only6,only7,only8,only9,only10,only13,only16,only17,only19,only20,only22,head \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --thresholds 0.4 \
  --timesteps 16 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn_relu_block_map_p99x04_t16.csv \
  --stats-out results/single_chunk/relu_activation_stats_manual.csv
```

Özet tablo:

| Scope | Replaced | Precision | Recall | Test mAP50 | Test mAP50-95 | Yorum |
|-------|----------|-----------|--------|------------|---------------|-------|
| only1 | 1 | 0.1234 | 0.2639 | 0.0744 | 0.0143 | kırıcı, eşik hassas |
| only2 | 4 | 0.4012 | 0.2917 | 0.1903 | 0.0812 | kırıcı |
| only3 | 1 | 0.9064 | 0.6729 | 0.7789 | 0.5253 | kısmi kayıp |
| only4 | 4 | 0.7804 | 0.5556 | 0.6054 | 0.4017 | kısmi kayıp |
| only5 | 1 | 0.8850 | 0.9306 | 0.9558 | 0.7340 | güvenli |
| only6 | 9 | 0.8877 | 0.8611 | 0.9412 | 0.7095 | güvenli |
| only7 | 1 | 0.8725 | 0.9501 | 0.9540 | 0.7341 | güvenli |
| only8 | 9 | 0.8530 | 0.9722 | 0.9523 | 0.7323 | güvenli |
| only9 | 1 | 0.8955 | 0.9522 | 0.9567 | 0.7351 | güvenli |
| only10 | 3 | 0.8808 | 0.9583 | 0.9547 | 0.7349 | güvenli |
| only13 | 9 | 0.8762 | 0.7862 | 0.9241 | 0.6651 | hafif kayıplı |
| only16 | 9 | 0.0600 | 0.0833 | 0.0259 | 0.0036 | kırıcı |
| only17 | 1 | 0.8952 | 0.9491 | 0.9559 | 0.7389 | güvenli |
| only19 | 9 | 0.8952 | 0.9491 | 0.9559 | 0.7389 | güvenli |
| only20 | 1 | 0.8952 | 0.9491 | 0.9559 | 0.7389 | güvenli |
| only22 | 5 | 0.8952 | 0.9491 | 0.9559 | 0.7389 | güvenli |
| head | 36 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | kırıcı |

Kümülatif aralık haritası:

```bash
python3 scripts/run_hybrid_snn.py \
  --checkpoint runs/single_chunk/chunk2_LF_relu/weights/best.pt \
  --scopes none,range1-2,range3-4,range5-10,range13-13,range16-16,range17-22,range3-10,range5-13,range3-13,range5-22 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --thresholds 0.4 \
  --timesteps 16 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn_relu_range_map_p99x04_t16.csv \
  --stats-out results/single_chunk/relu_activation_stats_manual.csv
```

Özet:

| Scope | Replaced | Precision | Recall | Test mAP50 | Test mAP50-95 |
|-------|----------|-----------|--------|------------|---------------|
| range1-2 | 5 | 0.0376 | 0.3194 | 0.0146 | 0.0024 |
| range3-4 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| range5-10 | 24 | 0.9598 | 0.6944 | 0.8769 | 0.6267 |
| range13-13 | 9 | 0.8762 | 0.7862 | 0.9241 | 0.6651 |
| range16-16 | 9 | 0.0600 | 0.0833 | 0.0259 | 0.0036 |
| range17-22 | 16 | 0.8952 | 0.9491 | 0.9559 | 0.7389 |
| range3-10 | 29 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| range5-13 | 33 | 0.9257 | 0.6667 | 0.8585 | 0.5447 |
| range3-13 | 38 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| range5-22 | 58 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Kırıcı bloklarda eşik duyarlılığı:

```bash
python3 scripts/run_hybrid_snn.py \
  --checkpoint runs/single_chunk/chunk2_LF_relu/weights/best.pt \
  --scopes only1,only2,range3-4,only16 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --thresholds 0.1,0.2,0.4,0.8 \
  --timesteps 16 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn_relu_sensitive_blocks_thresholds.csv \
  --stats-out results/single_chunk/relu_activation_stats_manual.csv
```

En iyi değerler:

| Scope | Best multiplier | Replaced | Precision | Recall | Test mAP50 | Test mAP50-95 |
|-------|-----------------|----------|-----------|--------|------------|---------------|
| only1 | 0.1 | 1 | 0.9609 | 0.6667 | 0.8214 | 0.4330 |
| only2 | 0.4 | 4 | 0.4012 | 0.2917 | 0.1903 | 0.0812 |
| range3-4 | 0.8 | 5 | 0.0512 | 0.0694 | 0.0070 | 0.0011 |
| only16 | 0.4 | 9 | 0.0600 | 0.0833 | 0.0259 | 0.0036 |

Yorum:

Bu harita, spiking dönüşümün bloklar arasında homojen davranmadığını açıkça
gösteriyor. `model.5-10` aralığı 24 aktivasyon spiking iken hâlâ makul
performans veriyor; `model.17-22` aralığı ise 16 aktivasyonla analog sonuçla
aynı kalıyor. Buna karşılık `model.2`, `model.3-4`, `model.16` ve detection
head doğrudan kırıcı bölgeler. `model.1` ise sabit 0.4 katsayısında kötü
görünse de 0.1 katsayısında toparlıyor; bu blok tamamen kırıcı değil, çok dar
eşik aralığına duyarlı. Bir sonraki aşamada mantıklı dönüşüm stratejisi,
erken blokları ve head'i analog bırakıp `model.5-10`, `model.13` ve
`model.17-22` gibi görece güvenli bölgeleri SNN-aware fine-tuning ile
birleştirmektir.

## Güvenli Blokların Birleşik Hybrid SNN Denemesi

Tekil ve aralık bazlı haritalardan sonra `scripts/run_hybrid_snn.py` içine
birleşik scope desteği eklendi. `+` işaretiyle birden fazla scope aynı
denemede birleştirilebiliyor:

- Örnek: `range5-10+range13-13+range17-22`
- Bu scope, `model.5-10`, `model.13` ve `model.17-22` bloklarını spiking
  yaparken `model.16` gibi kırıcı blokları ve detection head'i analog bırakır.

İlk birleşik tarama:

```bash
python3 scripts/run_hybrid_snn.py \
  --checkpoint runs/single_chunk/chunk2_LF_relu/weights/best.pt \
  --scopes none,range5-10,range13-13,range17-22,range5-10+range13-13,range5-10+range17-22,range13-13+range17-22,range5-10+range13-13+range17-22 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --thresholds 0.4 \
  --timesteps 16 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn_relu_safe_combo_p99x04_t16.csv \
  --stats-out results/single_chunk/relu_activation_stats_manual.csv
```

Sonuçlar:

| Scope | Replaced | Precision | Recall | Test mAP50 | Test mAP50-95 |
|-------|----------|-----------|--------|------------|---------------|
| none | 0 | 0.8952 | 0.9491 | 0.9559 | 0.7389 |
| range5-10 | 24 | 0.9598 | 0.6944 | 0.8769 | 0.6267 |
| range13-13 | 9 | 0.8762 | 0.7862 | 0.9241 | 0.6651 |
| range17-22 | 16 | 0.8952 | 0.9491 | 0.9559 | 0.7389 |
| range5-10+range13-13 | 33 | 0.9257 | 0.6667 | 0.8585 | 0.5447 |
| range5-10+range17-22 | 40 | 0.9598 | 0.6944 | 0.8769 | 0.6267 |
| range13-13+range17-22 | 25 | 0.8762 | 0.7862 | 0.9241 | 0.6651 |
| range5-10+range13-13+range17-22 | 49 | 0.9257 | 0.6667 | 0.8547 | 0.5428 |

Bu sonuç, kırıcı bloklar dışarıda bırakıldığında 49 aktivasyonun aynı anda
spiking yapılabildiğini gösteriyor. `range17-22` bloğunun eklenmesi, tek başına
analog sonucu bozmadığı gibi `range5-10` veya `range13` ile birleştiğinde de
ek kayıp getirmedi. Asıl performans kaybı `range13` eklendiğinde belirginleşiyor.

Üçlü güvenli kombinasyon için eşik taraması:

```bash
python3 scripts/run_hybrid_snn.py \
  --checkpoint runs/single_chunk/chunk2_LF_relu/weights/best.pt \
  --scopes range5-10+range13-13+range17-22 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --thresholds 0.2,0.3,0.4,0.5,0.6 \
  --timesteps 16 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn_relu_safe_combo_threshold_sweep.csv \
  --stats-out results/single_chunk/relu_activation_stats_manual.csv
```

Sonuç:

| Multiplier | Replaced | Precision | Recall | Test mAP50 | Test mAP50-95 |
|------------|----------|-----------|--------|------------|---------------|
| 0.2 | 49 | 0.9594 | 0.5417 | 0.8472 | 0.5678 |
| 0.3 | 49 | 0.9720 | 0.5139 | 0.8306 | 0.5260 |
| 0.4 | 49 | 0.9257 | 0.6667 | 0.8547 | 0.5428 |
| 0.5 | 49 | 0.9793 | 0.6579 | 0.8431 | 0.5578 |
| 0.6 | 49 | 0.9800 | 0.6815 | 0.8752 | 0.5795 |

Yorum:

Şu ana kadarki en güçlü geniş hybrid SNN sonucu, `range5-10+range13-13+range17-22`
scope'u ile `p99 * 0.6`, `T=16` ayarında elde edildi. Bu ayarda 49 aktivasyon
spiking, erken kırıcı bloklar ve detection head analog kalıyor:

- Test mAP50: 0.8752
- Test mAP50-95: 0.5795
- Analog ReLU referansına göre mAP50 korunumu: yaklaşık %91.6
- Analog ReLU referansına göre mAP50-95 korunumu: yaklaşık %78.4

Bu deney, blok duyarlılık haritasının yalnızca açıklayıcı değil, tasarım
seçimi için de kullanılabilir olduğunu gösteriyor. Sonraki mantıklı adım,
bu 49-aktivasyonluk hibrit mimari üzerinde SNN-aware fine-tuning veya
channel-wise threshold/scale kalibrasyonu denemektir.

## Split ve Modalite Doğrulaması

Güvenli blok birleşimi için en iyi ayar (`range5-10+range13-13+range17-22`,
`p99 * 0.6`, `T=16`) yalnızca LF test split'inde değil, LF validation ve HF
modalitesinde de denendi. Amaç, seçilen blok haritasının tek bir split'e veya
tek modaliteye özgü bir sonuç olup olmadığını kontrol etmekti.

Çalıştırılan dosyalar:

- LF val: `results/single_chunk/hybrid_snn_relu_safe_combo_lf_val.csv`
- HF val: `results/single_chunk/hybrid_snn_relu_safe_combo_hf_val.csv`
- HF test: `results/single_chunk/hybrid_snn_relu_safe_combo_hf_test.csv`
- HF ReLU baseline: `results/single_chunk/chunk2_HF_relu_metrics.txt`

Sonuçlar:

| Modality | Split | Model | Replaced | Precision | Recall | mAP50 | mAP50-95 |
|----------|-------|-------|----------|-----------|--------|-------|----------|
| LF | val | ReLU analog | 0 | 0.9348 | 0.9054 | 0.9310 | 0.6687 |
| LF | val | Hybrid SNN | 49 | 0.9735 | 0.7973 | 0.8751 | 0.5662 |
| HF | val | ReLU analog | 0 | 0.8882 | 0.7857 | 0.9263 | 0.5479 |
| HF | val | Hybrid SNN | 49 | 0.9181 | 0.8429 | 0.8966 | 0.5291 |
| HF | test | ReLU analog | 0 | 0.8787 | 0.7778 | 0.8694 | 0.5316 |
| HF | test | Hybrid SNN | 49 | 0.8750 | 0.8095 | 0.8543 | 0.5201 |

Korunum oranları:

| Modality | Split | mAP50 retention | mAP50-95 retention |
|----------|-------|-----------------|--------------------|
| LF | val | 94.0% | 84.7% |
| HF | val | 96.8% | 96.6% |
| HF | test | 98.3% | 97.8% |

Yorum:

LF testte bulunan güvenli blok kombinasyonu, LF val split'inde de çalıştı ve
HF modalitesinde çok daha yüksek metrik korunumu verdi. Bu önemli bir bulgu:
tam dönüşüm başarısız olurken, blok duyarlılık haritasına göre seçilen orta ve
geç bloklar farklı split/modalite koşullarında da spiking yapılabiliyor.
Özellikle HF val/test sonuçları, bu hibrit stratejinin yalnızca LF test'e
özgü olmadığını gösteriyor. Bir sonraki mantıklı adım, bu sabit 49
aktivasyonluk mimariyi SNN-aware fine-tuning veya daha ayrıntılı
threshold/scale kalibrasyonu ile iyileştirmektir.

### Fine-Tune Olmadan Daha Uzun Timestep Denemesi

LF ve HF modalitelerini ayrıştırmadan ortak bir protokol denemek için güvenli
blok kombinasyonu fine-tune yapılmadan `T=32` ile tekrar değerlendirildi. Bu
deneyde scope, threshold, calibration stat, splitler ve checkpointler aynı
tutuldu; yalnızca hybrid değerlendirme timestep sayısı `T=16` yerine `T=32`
yapıldı. Amaç, daha uzun temporal integration'ın iki modalitede de stabil bir
kazanım sağlayıp sağlamadığını ölçmekti.

Çalıştırılan komut:

```bash
python3 scripts/run_hybrid_batch.py \
  --dataset single \
  --mod both \
  --splits val,test \
  --activation relu \
  --tag chunk2 \
  --scopes none,range5-10+range13-13+range17-22 \
  --thresholds 0.6 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --timesteps 32 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn/batch_single_lf_hf_val_test_safe_t32.csv
```

Sonuçlar:

| Modality | Split | Model | T | Precision | Recall | mAP50 | mAP50-95 |
|----------|-------|-------|---|-----------|--------|-------|----------|
| LF | val | ReLU analog | 1 | 0.9348 | 0.9054 | 0.9310 | 0.6687 |
| LF | val | Hybrid SNN | 32 | 0.9692 | 0.8496 | 0.9141 | 0.6032 |
| LF | test | ReLU analog | 1 | 0.8952 | 0.9491 | 0.9559 | 0.7389 |
| LF | test | Hybrid SNN | 32 | 0.9024 | 0.7709 | 0.9221 | 0.6425 |
| HF | val | ReLU analog | 1 | 0.8882 | 0.7857 | 0.9263 | 0.5479 |
| HF | val | Hybrid SNN | 32 | 0.8778 | 0.8571 | 0.8942 | 0.5235 |
| HF | test | ReLU analog | 1 | 0.8787 | 0.7778 | 0.8694 | 0.5316 |
| HF | test | Hybrid SNN | 32 | 0.8769 | 0.7919 | 0.8533 | 0.5127 |

Yorum:

`T=32` denemesi fine-tune olmadan ortak ve güvenli bir iyileştirme üretmedi.
LF test split'inde `mAP50` `T=16` seviyesine göre yükselmiş görünse de
`mAP50-95` ve val performansı analogdan belirgin uzak kaldı. HF tarafında ise
`T=32`, önceki `T=16` hybrid inference sonucuna göre anlamlı bir kazanç
sağlamadı; HF test `mAP50-95` 0.5201 civarındaki `T=16` sonucundan 0.5127'ye
geriledi.

Bu sonuç, yalnızca timestep sayısını artırmanın LF/HF için ortak çözüm
olmadığını gösteriyor. Bundan sonra ortak metod geliştirme ekseni, daha uzun
temporal integration yerine threshold/scale kalibrasyonu, spiking residual blok
tasarımı veya teacher-student distillation gibi literatürde kullanılan daha
yapısal mekanizmalara yönelmelidir.

### Ortak Threshold Sweep Denemesi

LF ve HF modalitelerini ayrı ayrı optimize etmeden, tek bir ortak hybrid SNN
protokolü seçebilmek için güvenli blok kombinasyonu üzerinde threshold multiplier
sweep yapıldı. Scope, timestep, calibration stat ve veri splitleri aynı tutuldu;
yalnızca `p99` threshold çarpanı `0.4, 0.5, 0.6, 0.7, 0.8` değerlerinde
değiştirildi.

Çalıştırılan komut:

```bash
python3 scripts/run_hybrid_batch.py \
  --dataset single \
  --mod both \
  --splits val,test \
  --activation relu \
  --tag chunk2 \
  --scopes none,range5-10+range13-13+range17-22 \
  --thresholds 0.4,0.5,0.6,0.7,0.8 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --timesteps 16 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn/batch_single_lf_hf_val_test_safe_threshold_sweep_t16.csv
```

Hybrid sonuçları:

| Threshold | LF val mAP50-95 | LF test mAP50-95 | HF val mAP50-95 | HF test mAP50-95 | Avg mAP50-95 |
|-----------|-----------------|------------------|-----------------|------------------|--------------|
| 0.4 | 0.5112 | 0.5428 | 0.5449 | 0.5228 | 0.5304 |
| 0.5 | 0.5465 | 0.5578 | 0.5269 | 0.5195 | 0.5377 |
| 0.6 | 0.5662 | 0.5795 | 0.5291 | 0.5201 | 0.5487 |
| 0.7 | 0.5647 | 0.5690 | 0.4909 | 0.4568 | 0.5204 |
| 0.8 | 0.5362 | 0.5666 | 0.4989 | 0.4734 | 0.5188 |

En iyi split-bazlı değerler:

| Modality | Split | Best threshold | mAP50 | mAP50-95 |
|----------|-------|----------------|-------|----------|
| LF | val | 0.6 | 0.8751 | 0.5662 |
| LF | test | 0.6 | 0.8752 | 0.5795 |
| HF | val | 0.4 | 0.9368 | 0.5449 |
| HF | test | 0.4 | 0.8640 | 0.5228 |

Yorum:

Split-bazlı optimumlar modaliteye göre değişti: LF için `0.6`, HF için `0.4`
daha iyi göründü. Ancak LF/HF ayrıştırmadan tek bir ortak protokol seçme
hedefi açısından `0.6` en dengeli ayar oldu. Dört değerlendirme durumunun
ortalama `mAP50-95` değeri `0.6` threshold'da en yüksek çıktı ve en düşük
korunum oranı da diğer adaylara göre daha güvenli kaldı.

Bu nedenle mevcut tek-chunk çalışma için ortak hybrid inference protokolü şu
şekilde korunmalıdır: `range5-10+range13-13+range17-22`, `p99 * 0.6`, `T=16`,
`spike-scale=threshold`. HF'nin `0.4` eşikte daha iyi davranması önemli bir
modalite duyarlılığı bulgusudur; fakat tez anlatısını dağıtmamak için bu durum
ayrı bir HF reçetesi olarak değil, ortak yöntemin modaliteye duyarlılığı olarak
raporlanmalıdır.

### Channel-Wise Threshold Kalibrasyonu

Spiking-YOLO literatüründe kullanılan channel-wise normalization fikrine
yaklaşmak için global aktivasyon eşiği yerine her aktivasyon kanalına ayrı
`p99` threshold hesaplandı. Scope, timestep, checkpoint, veri splitleri ve
spike-scale aynı tutuldu; yalnızca kalibrasyon granülerliği `global` yerine
`channel` yapıldı. Bu deneme LF/HF için ayrı reçete üretmeden, aynı dönüşüm
metodunu iki modaliteye de uygulamak amacıyla yapıldı.

Çalıştırılan komut:

```bash
python3 scripts/run_hybrid_batch.py \
  --dataset single \
  --mod both \
  --splits val,test \
  --activation relu \
  --tag chunk2 \
  --scopes none,range5-10+range13-13+range17-22 \
  --thresholds 0.4,0.5,0.6,0.7,0.8 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --calibration-granularity channel \
  --spike-scale threshold \
  --timesteps 16 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn/batch_single_lf_hf_val_test_safe_channelwise_threshold_sweep_t16.csv
```

Hybrid sonuçları:

| Threshold | LF val mAP50-95 | LF test mAP50-95 | HF val mAP50-95 | HF test mAP50-95 | Avg mAP50-95 |
|-----------|-----------------|------------------|-----------------|------------------|--------------|
| 0.4 | 0.5995 | 0.6467 | 0.5413 | 0.5147 | 0.5756 |
| 0.5 | 0.6023 | 0.6322 | 0.5386 | 0.5040 | 0.5693 |
| 0.6 | 0.6035 | 0.6220 | 0.5312 | 0.5113 | 0.5670 |
| 0.7 | 0.6003 | 0.6187 | 0.5324 | 0.5035 | 0.5637 |
| 0.8 | 0.5931 | 0.6194 | 0.5306 | 0.4946 | 0.5594 |

En iyi split-bazlı değerler:

| Modality | Split | Best threshold | mAP50 | mAP50-95 |
|----------|-------|----------------|-------|----------|
| LF | val | 0.6 | 0.8898 | 0.6035 |
| LF | test | 0.4 | 0.8729 | 0.6467 |
| HF | val | 0.4 | 0.9367 | 0.5413 |
| HF | test | 0.4 | 0.8679 | 0.5147 |

Yorum:

Channel-wise kalibrasyon, global threshold sweep'e göre ortak ortalama
performansı yükseltti. Global kalibrasyonda en iyi ortak ortalama `mAP50-95`
`0.5487` iken channel-wise kalibrasyonda en iyi ortak ortalama `0.5756` oldu.
Bu artış özellikle LF tarafında belirgin: LF test `mAP50-95`, global ortak
protokolde `0.5795` iken channel-wise `0.4` eşikte `0.6467` seviyesine çıktı.

HF tarafında tablo daha temkinli okunmalı. HF val ve test analog baseline'a
yakın kalıyor fakat channel-wise yöntem global threshold'a göre HF için ek bir
kazanç üretmedi; HF test `mAP50-95` global sweep'teki `0.5228` seviyesinden
`0.5147` seviyesine indi. Bu nedenle channel-wise kalibrasyon ortak protokol
adayını güçlendirse de HF modalitesindeki kırılganlığı tamamen çözmedi.

Mevcut tek-chunk bulgusuna göre literatürle uyumlu en mantıklı ortak inference
adayı şudur: `range5-10+range13-13+range17-22`, channel-wise `p99 * 0.4`,
`T=16`, `spike-scale=threshold`. Bu sonuç nihai iddia değildir; full veri veya
k-fold protokole taşınmadan önce küçük veri üzerinde metod seçimi için
kullanılmalıdır.

### Ana Tez Tablosu: YOLOv26n-ANN ve YOLOv26n-SNN Adayı

Deneylerin kalabalık görünmemesi için mevcut tek-chunk sonuçlarından sade bir
ana karşılaştırma tablosu çıkarıldı. Burada ANN satırı ReLU ile eğitilmiş
YOLOv26n baseline'ı, SNN satırı ise seçilen ortak hybrid SNN adayını temsil
eder: `range5-10+range13-13+range17-22`, channel-wise `p95 * 0.4`, `T=16`,
`spike-scale=threshold`.

Bu tablo, ana train-then-convert protokolünü tek komutla çalıştıran yeni script
ile üretildi:

```bash
python3 scripts/run_train_then_convert_snn.py \
  --dataset single \
  --mod both \
  --splits val,test \
  --activation relu \
  --tag chunk2 \
  --snn-scope range5-10+range13-13+range17-22 \
  --snn-threshold 0.4 \
  --timesteps 16 \
  --source-activation ReLU \
  --calibration-stat p95 \
  --calibration-granularity channel \
  --spike-scale threshold \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out-dir results/single_chunk/train_then_convert_snn_p95_channel_t16
```

Çıktı:

```text
results/single_chunk/train_then_convert_snn_p95_channel_t16/single_ann_vs_snn_comparison.csv
results/single_chunk/train_then_convert_snn_p95_channel_t16/single_train_then_convert_snn_aggregate.csv
```

Sonuç:

| Modality | Split | ANN mAP50 | SNN mAP50 | ANN mAP50-95 | SNN mAP50-95 | mAP50-95 retention |
|----------|-------|-----------|-----------|--------------|--------------|--------------------|
| LF | val | 0.9310 | 0.8974 | 0.6687 | 0.6032 | 90.22% |
| LF | test | 0.9559 | 0.8815 | 0.7389 | 0.6517 | 88.20% |
| HF | val | 0.9263 | 0.9354 | 0.5479 | 0.5423 | 98.97% |
| HF | test | 0.8694 | 0.8765 | 0.5316 | 0.5148 | 96.83% |

Yorum:

Tezin ana anlatısı için en temiz sonuç budur: YOLOv26n tabanlı ANN baseline ile
YOLOv26n tabanlı SNN/hybrid-SNN adayının aynı veri ve splitlerde karşılaştırması.
HF tarafında SNN adayı ANN performansını neredeyse korudu; LF tarafında daha
belirgin kayıp var fakat model tamamen kırılmadan çalışır durumda kaldı.

Bu ana aday, önceki `p99 * 0.4, T=16` ve `p99 * 0.4, T=32` varyantlarıyla
karşılaştırılarak seçildi. Dört değerlendirme durumunun ortalama SNN
`mAP50-95` değeri `p99,T=16` için `0.5756`, `p99,T=32` için `0.5771`,
`p95,T=16` için `0.5780` oldu. `T=32`, LF'de küçük kazanç sağlasa da HF testte
geriledi ve inference maliyetini iki katına çıkardı. Bu nedenle mevcut pilot
çalışmada en dengeli ve maliyet-etkin ortak protokol channel-wise `p95 * 0.4`,
`T=16` olarak güncellendi.

Bu nedenle tez metninde ana omurga şu şekilde sade tutulabilir: önce ANN
YOLOv26n baseline, sonra seçilen YOLOv26n-SNN adayı, ardından performans
korunumu. Diğer threshold, timestep, fine-tune ve native pilot denemeleri ana
sonucu kalabalıklaştırmak yerine bu adayın nasıl seçildiğini ve tam SNN'ye
geçişte nerelerin zor olduğunu açıklayan ablation/tartışma bölümü olarak
kullanılmalıdır.

### Metodolojik Karar: Train-Then-Convert SNN Rotası

Bu aşamada ana metodoloji sıfırdan SNN eğitmek değil, literatürdeki
Spiking-YOLO çizgisine daha yakın olan train-then-convert yaklaşımı olarak
sabitlenmelidir. Yani önce YOLOv26n ANN modeli boru tespiti için başarılı
şekilde eğitilir; ardından model özel dönüşüm mekanizmalarıyla SNN/hybrid-SNN
forma taşınır.

Bu kararın gerekçesi:

- Nesne tespitinde bbox regresyonu, sınıflandırmaya göre daha hassastır; ham
  aktivasyon değişimi modeli kolayca sıfıra düşürebilir.
- Literatürde başarılı SNN detector örnekleri genellikle yalnızca aktivasyonları
  değiştirmez; channel-wise normalization, threshold/scale kalibrasyonu, signed
  neuron, residual-spiking bloklar veya distillation gibi ek mekanizmalar
  kullanır.
- Bizim pilot sonuçlarımız da aynı davranışı gösterdi: ham `ReLU -> IFNode`
  dönüşümü kırılgan, fakat channel-wise threshold/scale kalibrasyonu ile çalışan
  bir YOLOv26n-SNN adayı elde edilebiliyor.
- Sıfırdan full-SNN eğitmek, bu tez için ana deney yolunu gereksiz
  risklendirebilir. Bunun yerine ANN'den güçlü bir başlangıç alıp dönüşüm
  kalitesini artırmak daha kontrollü ve literatürle uyumlu bir yöntemdir.

Bu nedenle tezde ana yöntem şu şekilde ifade edilebilir:

> Önce su altı boru tespiti için YOLOv26n tabanlı ANN model eğitilmiş, ardından
> seçili feature extraction blokları channel-wise kalibre edilmiş IF spiking
> node'lara dönüştürülerek YOLOv26n tabanlı SNN/hybrid-SNN detector elde
> edilmiştir.

Burada "hybrid" ifadesi özellikle dürüst tutulmalıdır: detection head analog
bırakıldığı için model tam full-SNN değildir. Ancak spiking dönüşüm feature
extraction katmanlarında uygulandığı, timestep tabanlı inference kullanıldığı ve
spiking node'lar channel-wise kalibre edildiği için yöntem SNN tabanlı nesne
tespit yaklaşımı olarak savunulabilir. Full veri geldiğinde ana hedef, bu
train-then-convert protokolünü k-fold ölçekte çalıştırmak ve ANN'e göre
performans korunumunu raporlamaktır.

### Full/Backbone Dönüşüm Kontrolü

Seçili blok protokolünün neden gerekli olduğunu doğrulamak için aynı
channel-wise `p95 * 0.4`, `T=16` kalibrasyonu ile daha agresif iki kapsam test
edildi: `backbone` ve `all`. Bu deney, ANN eğitildikten sonra modelin tek
adımda tamamen veya neredeyse tamamen spiking forma taşınıp taşınamayacağını
kontrol etmek için yapıldı.

Çıktılar:

```text
results/single_chunk/hybrid_snn/full_scope_lf_p95_channel_t16.csv
results/single_chunk/hybrid_snn/full_scope_hf_p95_channel_t16.csv
```

Sonuç:

| Modality | Scope | Replaced | Test mAP50 | Test mAP50-95 |
|----------|-------|----------|------------|---------------|
| LF | backbone | 69 | 0.0000 | 0.0000 |
| LF | all | 105 | 0.0000 | 0.0000 |
| HF | backbone | 69 | 0.0000 | 0.0000 |
| HF | all | 105 | 0.0000 | 0.0000 |

Yorum:

Bu sonuç, YOLOv26n ANN modelinin tek adımda backbone veya full-SNN forma
dönüştürülemediğini gösteriyor. Daha düşük `p95` threshold ve channel-wise
kalibrasyon bile full kapsamda yeterli olmadı. Dolayısıyla uygulanabilir yol,
modeli bir anda tamamen spiking yapmak değil; önce çalışan seçili blok
dönüşümünü kullanmak, ardından hassas blokları kademeli şekilde ekleyip
fine-tune/distillation gibi ek mekanizmalarla toparlamaya çalışmaktır.

## Pilot Chunk'tan K-Fold Protokole Taşıma

Mevcut SNN denemeleri bilinçli olarak tek chunk/small-run üzerinde yürütüldü.
Bu tercih, tam 5-fold deneyleri başlatmadan önce dönüşümün nerede kırıldığını
hızlıca haritalamak içindi. Tez için nihai iddia tek chunk sonucuna
dayandırılmayacak; aynı ayarların fold bazlı ANN/SNN karşılaştırmasına
taşınması gerekecek.

Bu nedenle scriptler taşınabilir hale getirildi:

- `scripts/run_single_chunk.py` artık `--tag`, `--dataset-dir` ve
  `--run-group` alıyor. Varsayılan davranış hâlâ `chunk2_*` run'larını üretir,
  fakat aynı script başka pilot chunk veya küçük dataset için de kullanılabilir.
- `scripts/run_hybrid_batch.py` eklendi. Bu script aynı hybrid SNN ayarını
  `single` veya `kfold` dataset düzeninde seri olarak çalıştırır ve aggregate
  CSV üretir.

Tek chunk pilot doğrulama örneği:

```bash
python3 scripts/run_hybrid_batch.py \
  --dataset single \
  --tag chunk2 \
  --mod HF \
  --splits test \
  --activation relu \
  --scopes none,range5-10+range13-13+range17-22 \
  --thresholds 0.6 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --timesteps 16 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --out results/single_chunk/hybrid_snn/batch_single_hf_test_smoke.csv
```

Bu smoke test, HF test split'inde önceki sonucu yeniden verdi:

| Model | Replaced | Test mAP50 | Test mAP50-95 |
|-------|----------|------------|---------------|
| ReLU analog | 0 | 0.8694 | 0.5316 |
| Hybrid SNN | 49 | 0.8543 | 0.5201 |

K-fold aşamasına geçildiğinde beklenen kullanım:

```bash
python3 scripts/run_hybrid_batch.py \
  --dataset kfold \
  --folds 0 1 2 3 4 \
  --mod both \
  --splits val,test \
  --activation relu \
  --scopes none,range5-10+range13-13+range17-22 \
  --thresholds 0.6 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --timesteps 16 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --skip-missing
```

Yorum:

Tek chunk pilot sonuçları yöntem geliştirme ve ablation için kullanılacak.
Asıl genelleme iddiası ise aynı scriptlerin üreteceği fold bazlı sonuçlarla
kurulacak. Böylece bugün elde edilen güvenli blok haritası kaybolmadan,
ileride 5 chunk protokolüne düzenli biçimde taşınabilecek.

## İlk SNN-Aware Fine-Tuning Denemesi

Güvenli blok kombinasyonu için ilk kısa SNN-aware fine-tuning denemesi LF
single-chunk üzerinde yapıldı. Bu denemede aynı 49 aktivasyon spiking kaldı;
erken kırıcı bloklar ve detection head analog bırakıldı. Eğitim sırasında
`T=4`, değerlendirme sırasında önceki karşılaştırmalarla uyumlu olacak şekilde
`T=16` kullanıldı.

Bu deneme için `scripts/run_hybrid_snn.py` eğitim yolunu destekleyecek şekilde
genişletildi:

- `HybridSNNModel` artık `dict` batch girdisinde Ultralytics loss yolunu
  çağırabiliyor.
- Timestep ortalaması tensor, list, tuple ve dict çıktıları için çalışıyor.
- `criterion`, `args`, `names`, `nc` gibi Ultralytics trainer/loss
  beklentileri wrapper ile iç `base` model arasında aktarılıyor.

Ek olarak `scripts/run_hybrid_snn_finetune.py` eklendi. Bu script:

- aktivasyon istatistiklerini toplar,
- güvenli blokları spiking yapar,
- pre-finetune val/test metriklerini alır,
- custom `DetectionTrainer` ile hybrid modeli eğitir,
- post-finetune val/test metriklerini CSV'ye yazar.

Çalıştırılan komut:

```bash
python3 scripts/run_hybrid_snn_finetune.py \
  --checkpoint runs/single_chunk/chunk2_LF_relu/weights/best.pt \
  --data datasets/subpipe_single_chunk/LF/data.yaml \
  --scope range5-10+range13-13+range17-22 \
  --threshold 0.6 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --train-timesteps 4 \
  --eval-timesteps 16 \
  --epochs 3 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --lr0 0.0001 \
  --name chunk2_LF_relu_hybrid_snn_ft_t4_e3 \
  --out results/single_chunk/hybrid_snn_finetune_lf_t4_e3.csv \
  --stats-out results/single_chunk/hybrid_snn_finetune_lf_t4_e3_activation_stats.csv
```

Sonuçlar:

| Split | Stage | Precision | Recall | mAP50 | mAP50-95 |
|-------|-------|-----------|--------|-------|----------|
| val | pre-ft | 0.9750 | 0.7973 | 0.8717 | 0.5680 |
| val | post-ft | 0.9529 | 0.8199 | 0.8799 | 0.6053 |
| test | pre-ft | 0.9800 | 0.6815 | 0.8742 | 0.5749 |
| test | post-ft | 0.9425 | 0.6827 | 0.8320 | 0.6232 |

Yorum:

Bu ilk kısa fine-tuning denemesi val split'inde her iki mAP metriğini de
iyileştirdi: mAP50 0.8717'den 0.8799'a, mAP50-95 ise 0.5680'den 0.6053'e
çıktı. Test split'inde ise daha karmaşık bir davranış görüldü: mAP50 0.8742'den
0.8320'ye düştü, fakat mAP50-95 0.5749'dan 0.6232'ye yükseldi. Bu, kısa
SNN-aware fine-tuning'in daha sıkı IoU eşiklerinde kutu kalitesini
iyileştirebildiğini, ancak geniş mAP50/precision-recall dengesini bozabildiğini
gösteriyor.

Bu sonuç doğrudan nihai iyileştirme olarak değil, eğitim yolunun çalıştığını
kanıtlayan ve optimizasyon alanını açan ilk deney olarak yorumlanmalı. Sonraki
mantıklı varyasyonlar daha düşük LR, daha az augmentation, `T=8` eğitim veya
yalnızca spiking blok parametrelerini eğitme şeklinde olabilir.

### Augmentation Kapalı Fine-Tuning Ablasyonu

İlk fine-tuning sonucunda test split'inde `mAP50` düşerken `mAP50-95`
yükselmişti. Bu davranışın SNN-aware fine-tuning dinamiğinden mi yoksa küçük
single-chunk veri üzerinde agresif augmentation'dan mı kaynaklandığını ayırmak
için tek değişkenli bir ablation yapıldı. Model, scope, threshold, timestep,
epoch, batch, LR ve veri splitleri aynı tutuldu; yalnızca detection training
augmentation'ları kapatıldı.

Bu amaçla `scripts/run_hybrid_snn_finetune.py` içine `--augmentation
{default,none}` seçeneği eklendi. `none` modunda `mosaic`, `mixup`, `cutmix`,
`copy_paste`, geometrik dönüşümler, HSV değişimleri, flip ve erasing sıfırlandı.

Çalıştırılan komut:

```bash
python3 scripts/run_hybrid_snn_finetune.py \
  --checkpoint runs/single_chunk/chunk2_LF_relu/weights/best.pt \
  --data datasets/subpipe_single_chunk/LF/data.yaml \
  --scope range5-10+range13-13+range17-22 \
  --threshold 0.6 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --train-timesteps 4 \
  --eval-timesteps 16 \
  --epochs 3 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --lr0 0.0001 \
  --augmentation none \
  --name chunk2_LF_relu_hybrid_snn_ft_t4_e3_noaug \
  --out results/single_chunk/hybrid_snn_finetune_lf_t4_e3_noaug.csv \
  --stats-out results/single_chunk/hybrid_snn_finetune_lf_t4_e3_noaug_activation_stats.csv
```

Sonuç karşılaştırması:

| Augmentation | Split | Stage | Precision | Recall | mAP50 | mAP50-95 |
|--------------|-------|-------|-----------|--------|-------|----------|
| default | val | post-ft | 0.9529 | 0.8199 | 0.8799 | 0.6053 |
| default | test | post-ft | 0.9425 | 0.6827 | 0.8320 | 0.6232 |
| none | val | post-ft | 0.9435 | 0.9018 | 0.9295 | 0.6593 |
| none | test | post-ft | 0.8820 | 0.9347 | 0.9549 | 0.7146 |

Yorum:

Augmentation kapalı fine-tuning, önceki default-augmentation fine-tune'a göre
hem val hem test split'inde daha iyi sonuç verdi. Özellikle test split'inde
`mAP50` 0.8320'den 0.9549'a, `mAP50-95` ise 0.6232'den 0.7146'ya yükseldi.
Bu tek değişkenli karşılaştırma, önceki test `mAP50` düşüşünün büyük ölçüde
küçük veri üzerinde uygulanan augmentation rejimiyle ilişkili olabileceğini
gösteriyor.

Bu ayar, LF analog ReLU test sonucundaki `mAP50=0.9559` seviyesini neredeyse
geri kazanırken `mAP50-95=0.7389` analog sonucunun biraz altında kaldı. Bu
nedenle sonuç, hybrid SNN performansının ciddi biçimde toparlandığını fakat
lokalizasyon kalitesinde analog modelle arada hâlâ küçük bir fark bulunduğunu
gösteriyor.

Bu sonuç, SNN-aware fine-tuning'in güvenli spiking blok kombinasyonunda
performans toparlayabildiğine dair en güçlü pilot bulgudur. Ancak tek chunk
üzerinde elde edildiği için genelleme iddiası olarak değil, kfold/full-dataset
deneylerine taşınacak aday ayar olarak değerlendirilmelidir.

### HF Modalitesinde Aynı Ablation

Aynı augmentation-kapalı fine-tuning protokolü HF single-chunk modeli için de
çalıştırıldı. Bu deneyde yalnızca checkpoint ve data YAML HF olarak değiştirildi;
scope, threshold, timestep, epoch, batch, LR ve augmentation modu aynı tutuldu.

Çalıştırılan komut:

```bash
python3 scripts/run_hybrid_snn_finetune.py \
  --checkpoint runs/single_chunk/chunk2_HF_relu/weights/best.pt \
  --data datasets/subpipe_single_chunk/HF/data.yaml \
  --scope range5-10+range13-13+range17-22 \
  --threshold 0.6 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --train-timesteps 4 \
  --eval-timesteps 16 \
  --epochs 3 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --lr0 0.0001 \
  --augmentation none \
  --name chunk2_HF_relu_hybrid_snn_ft_t4_e3_noaug \
  --out results/single_chunk/hybrid_snn_finetune_hf_t4_e3_noaug.csv \
  --stats-out results/single_chunk/hybrid_snn_finetune_hf_t4_e3_noaug_activation_stats.csv
```

LF ve HF sonuçlarının aynı protokol altındaki karşılaştırması:

| Modality | Split | Stage | Precision | Recall | mAP50 | mAP50-95 |
|----------|-------|-------|-----------|--------|-------|----------|
| LF | val | pre-ft | 0.9750 | 0.7973 | 0.8717 | 0.5680 |
| LF | val | post-ft | 0.9435 | 0.9018 | 0.9295 | 0.6593 |
| LF | test | pre-ft | 0.9800 | 0.6815 | 0.8742 | 0.5749 |
| LF | test | post-ft | 0.8820 | 0.9347 | 0.9549 | 0.7146 |
| HF | val | pre-ft | 0.9181 | 0.8429 | 0.8959 | 0.5287 |
| HF | val | post-ft | 0.7834 | 0.8000 | 0.8487 | 0.5037 |
| HF | test | pre-ft | 0.8750 | 0.8095 | 0.8545 | 0.5198 |
| HF | test | post-ft | 0.7589 | 0.8095 | 0.8154 | 0.4696 |

Yorum:

Aynı augmentation-kapalı fine-tuning protokolü LF'de belirgin toparlanma
sağlarken HF'de performansı düşürdü. HF test split'inde `mAP50` 0.8545'ten
0.8154'e, `mAP50-95` ise 0.5198'den 0.4696'ya geriledi. Bu sonuç, güvenli
spiking blok kombinasyonunun inference seviyesinde HF için çalışmasına rağmen
fine-tuning dinamiğinin modaliteye duyarlı olduğunu gösteriyor.

Tez açısından önemli bulgu şudur: SNN-aware fine-tuning tek bir global reçete
gibi davranmıyor. LF için augmentation kapalı kısa fine-tune çok güçlü bir aday
ayar üretirken, HF için daha korumacı bir strateji denenmeli. Mantıklı HF
varyasyonları daha düşük LR, daha kısa epoch, yalnızca spiking blokları eğitme
veya fine-tune yerine sadece calibrated hybrid inference sonucunu koruma
şeklinde tasarlanabilir.

### HF Icin Daha Dusuk LR Denemesi

HF modalitesinde fine-tune performansi dustugu icin ayni protokol daha dusuk
learning rate ile tekrarlandi. Bu denemede tek degisken `lr0=0.00001` oldu;
augmentation kapali, scope, threshold, timestep ve epoch sayisi ayni tutuldu.

Calistirilan komut:

```bash
python3 scripts/run_hybrid_snn_finetune.py \
  --checkpoint runs/single_chunk/chunk2_HF_relu/weights/best.pt \
  --data datasets/subpipe_single_chunk/HF/data.yaml \
  --scope range5-10+range13-13+range17-22 \
  --threshold 0.6 \
  --source-activation ReLU \
  --calibration-stat p99 \
  --spike-scale threshold \
  --train-timesteps 4 \
  --eval-timesteps 16 \
  --epochs 3 \
  --imgsz 320 \
  --batch 4 \
  --device cpu \
  --lr0 0.00001 \
  --augmentation none \
  --name chunk2_HF_relu_hybrid_snn_ft_t4_e3_noaug_lr1e5 \
  --out results/single_chunk/hybrid_snn_finetune_hf_t4_e3_noaug_lr1e5.csv \
  --stats-out results/single_chunk/hybrid_snn_finetune_hf_t4_e3_noaug_lr1e5_activation_stats.csv
```

Sonuc:

| HF setting | Split | Stage | Precision | Recall | mAP50 | mAP50-95 |
|------------|-------|-------|-----------|--------|-------|----------|
| lr=1e-4 | val | post-ft | 0.7834 | 0.8000 | 0.8487 | 0.5037 |
| lr=1e-4 | test | post-ft | 0.7589 | 0.8095 | 0.8154 | 0.4696 |
| lr=1e-5 | val | post-ft | 0.6764 | 0.7167 | 0.7299 | 0.3533 |
| lr=1e-5 | test | post-ft | 0.7685 | 0.6984 | 0.7056 | 0.3499 |

Yorum:

HF icin learning rate'i 10 kat dusurmek performansi toparlamadi; aksine
`lr=1e-4` denemesinden de daha kotu sonuc verdi. Pre-finetune HF hybrid test
sonucu `mAP50=0.8545`, `mAP50-95=0.5198` iken `lr=1e-5` post-finetune sonucu
`mAP50=0.7056`, `mAP50-95=0.3499` seviyesine dustu. Bu, HF icin sorunun sadece
guncelleme adiminin buyuk olmasi olmadigini gosteriyor.

Bu noktada HF icin en guvenli sonuc calibrated hybrid inference sonucunu
korumaktir. Yeni HF denemeleri, yalnizca LR/epoch oynamak yerine spiking
bloklarin egitimini sinirlama, erken durdurma, katman dondurma veya literaturde
onerilen spiking-residual/zaman-normalizasyonu tasarimlarina yaklasma
seklinde planlanmalidir.

## SNN-Native Pilot: Surrogate Gradient ve Öğrenilebilir Threshold

Conversion tabanlı hibrit hattın sınırlarını gördükten sonra, ANN'den daha
fazla kopan bir pilot deneme yapıldı. Bu denemede channel-wise kalibrasyon
korundu, ancak channel-wise spiking node artık hard threshold yerine
surrogate-gradient kullanan bir ileri/geri yayılım davranışına geçirildi.
Ayrıca threshold değerleri fine-tune sırasında öğrenilebilir parametre olarak
açıldı. Detection head analog bırakıldı; böylece daha cesur spiking
backbone/neck denemesi yapılırken regresyon head'i tamamen spiking hale
getirilmedi.

Kod tarafındaki ekleme:

- `ChannelScaledIFNode`, surrogate sigmoid gradient ile çalışacak şekilde
  güncellendi.
- `run_hybrid_snn_finetune.py` içine `--calibration-granularity channel`,
  `--surrogate-alpha` ve `--learn-thresholds` seçenekleri eklendi.

### Full Backbone Spiking Pilot

İlk olarak tüm backbone'u spiking hale getiren agresif bir pilot denendi:
`scope=backbone`, channel-wise `p99 * 0.4`, `train_T=2`, `eval_T=8`,
`epochs=1`, `lr0=5e-5`, augmentation kapalı.

Çıktı:

```text
results/single_chunk/hybrid_snn/native_probe_lf_backbone_channel_t2_e1.csv
```

Sonuç:

| Modality | Scope | Replaced | Pre test mAP50-95 | Post test mAP50-95 |
|----------|-------|----------|-------------------|--------------------|
| LF | backbone | 69 | 0.0000 | 0.0000 |

Yorum:

Tam backbone dönüşümü LF'de bile 1 epoch surrogate fine-tune ile toparlanmadı.
Bu sonuç, erken backbone katmanlarının doğrudan spiking hale getirilmesinin
mevcut YOLO ağırlıklarıyla çok kırılgan olduğunu gösteriyor. Bu nedenle tam
SNN yönüne geçiş doğrudan tüm backbone'u dönüştürerek değil, hassas blokları
kademeli ekleyerek yapılmalıdır.

### Blok 16 Kırılma Noktası

Güvenli channel-wise protokol `range5-10+range13-13+range17-22` idi. Daha
fazla spiking kapsamına geçmek için `14`, `15` ve `16` blokları tek tek bu
güvenli kapsama eklendi.

Çıktı:

```text
results/single_chunk/hybrid_snn/native_probe_lf_safe_plus_each_14_16_channel_t16.csv
```

Sonuç:

| Scope | Replaced | LF test mAP50 | LF test mAP50-95 |
|-------|----------|---------------|------------------|
| safe + only14 | 49 | 0.8729 | 0.6467 |
| safe + only15 | 49 | 0.8729 | 0.6467 |
| safe + only16 | 58 | 0.0000 | 0.0000 |

Yorum:

`only14` ve `only15` ek aktivasyon değiştirmediği için güvenli protokolle aynı
kaldı. `only16` ise 9 ek aktivasyon dönüştürdü ve LF test sonucunu doğrudan
sıfıra indirdi. Bu bulgu, mevcut mimaride `model.16` çevresinin spiking
dönüşüm için kritik kırılma noktalarından biri olduğunu gösteriyor.

### Blok 16 Surrogate Fine-Tune

Ardından aynı genişletilmiş kapsam LF ve HF için aynı ayarla fine-tune edildi:
`range5-10+range13-13+range17-22+only16`, channel-wise `p99 * 0.4`,
learnable thresholds, `train_T=2`, `eval_T=16`, `epochs=1`, `lr0=5e-5`,
augmentation kapalı.

Çıktılar:

```text
results/single_chunk/hybrid_snn/native_probe_lf_safe_plus16_channel_ft_t2_e1.csv
results/single_chunk/hybrid_snn/native_probe_hf_safe_plus16_channel_ft_t2_e1.csv
```

Sonuç:

| Modality | Stage | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 |
|----------|-------|-----------|--------------|------------|---------------|
| LF | pre-ft | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| LF | post-ft | 0.7437 | 0.4427 | 0.7098 | 0.4365 |
| HF | pre-ft | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| HF | post-ft | 0.2907 | 0.0477 | 0.2254 | 0.0392 |

Yorum:

Bu deney conversion hattından farklı bir bulgu verdi. `only16` eklenince ham
inference tamamen sıfırlanmasına rağmen surrogate-gradient ve öğrenilebilir
threshold ile LF tarafında anlamlı bir geri kazanım oluştu. Bu, bazı kırılgan
spiking blokların sadece kalibrasyonla değil, eğitim sırasında yeniden
uyarlanarak kısmen kullanılabileceğini gösteriyor.

HF tarafında aynı protokol çok daha zayıf kaldı. Pre-finetune sıfırdan çıktı
ama final `eval_T=16` test sonucu yalnızca `mAP50-95=0.0392` oldu. Bu sonuç,
HF modalitesinde native/surrogate eğitimin daha dikkatli tasarım gerektirdiğini
gösteriyor: daha uzun eğitim, farklı timestep eşleşmesi, teacher-student
distillation, spiking residual blok veya timestep-aware normalization gibi
literatür mekanizmaları denenmeden HF için güçlü sonuç beklenmemelidir.

Bu pilotun tez açısından değeri, tam SNN yönünün bir anda tüm backbone'u
dönüştürerek değil, kırılma bloklarını tek tek bulup surrogate eğitimle
toparlanabilirliklerini ölçerek ilerlemesi gerektiğini göstermesidir.

## SNN Object Detection Literaturunden Yol Haritasi

Kisa literatur okumasi, SNN tabanli nesne tespitinde basarili yontemlerin
yalnizca mevcut YOLO aktivasyonlarini spiking node ile degistirmekten daha
fazlasini yaptigini gosteriyor:

- Spiking-YOLO, ANN-SNN donusumunu channel-wise normalization ve signed neuron
  gibi ozel dengeleme mekanizmalariyla stabilize ediyor. Bu, detection gibi
  regresyon agirlikli ciktilarda basit esikleme/donusumun yetmedigini gosteriyor.
  Kaynak: https://arxiv.org/abs/1903.06530
- EMS-YOLO, donusum yerine dogrudan surrogate-gradient ile egitilen full-spike
  residual bloklar tasarliyor ve 4 timestep gibi dusuk gecikmeyle calisabiliyor.
  Bu, bizim `T=4` train denemelerimize yakin ama mimari olarak daha ozel bir
  cizgi. Kaynak: https://arxiv.org/abs/2307.11411
- Spiking CenterNet, event-data detection icin spiking mimariyi ogretmen modelden
  distillation ile destekliyor. Bu, yalnizca detection loss ile spiking modeli
  itmenin zor oldugunu ve teacher supervision'in faydali olabilecegini gosteriyor.
  Kaynak: https://arxiv.org/abs/2402.01287
- SU-YOLO, underwater object detection icin spiking denoising, separated batch
  normalization ve spiking CSP/residual bloklar kullaniyor. Bu calisma sonar/
  sualti domainine yakin oldugu icin tez acisindan ozellikle onemli. Kaynak:
  https://arxiv.org/abs/2503.24389

Bu literatur isiginda bizim bulgumuz anlamli: LF'de guvenli bloklarin
augmentation kapali kisa fine-tune ile toparlanmasi, hybrid SNN donusumunun
potansiyelini gosteriyor. HF'de fine-tune'un bozulmasi ise mevcut wrapper
yaklasiminin her modalite icin yeterli olmadigini, daha ozel spiking blok
tasarimi veya distillation/zaman-normalizasyonu gerektirebilecegini gosteriyor.

Bir sonraki metodolojik adim olarak en makul iki rota:

1. Mevcut single-chunk veri uzerinde LF/HF birlikte raporlanan ortak protokolu
   sabitlemek ve yeni denemeleri iki modaliteye de ayni sekilde uygulamak.
2. Fine-tune/LR/timestep oynamak yerine literaturle uyumlu yeni bir spiking blok
   veya kalibrasyon denemesi yapmak: ornegin safe blocklarda spiking residual
   davranisi, timestep-aware BN/threshold kaydi veya teacher-student
   distillation.
