# SubPipe Tez Deneyleri

Bu repo, SubPipe side-scan sonar veri setinde tek sınıflı `pipe` tespiti için
YOLO26n tabanlı ANN baseline deneylerini ve 5-fold chunk-based cross validation
akışını içerir.

## Dataset Yerleşimi

Orijinal SubPipe klasörü git'e eklenmez. Her makinede lokal olarak aşağıdaki
konumlardan birine koyun:

```text
datasets/SubPipe/
```

veya:

```text
datasets/subpipe/
```

Alternatif olarak repo dışındaki bir dataset için `SUBPIPE_ROOT` ortam
değişkenini kullanabilirsiniz:

```bash
export SUBPIPE_ROOT=/path/to/SubPipe
```

Beklenen chunk yapısı:

```text
datasets/SubPipe/
├── Chunk0/
├── Chunk1/
├── Chunk2/
├── Chunk3/
└── Chunk4/
```

## Ana Komutlar

Aşağıdaki komutları repo kökünden çalıştırın:

```bash
python3 scripts/convert_pbm_to_png.py
python3 scripts/dataset_analysis.py
python3 scripts/prepare_kfold.py
python3 scripts/verify_kfold.py
python3 scripts/run_kfold.py
python3 scripts/kfold_analysis.py
```

`prepare_kfold.py`, `datasets/subpipe_kfold` altındaki YOLO dataset YAML'larını
repo taşınabilir olacak şekilde üretir ve image/label symlinklerini göreli
hedeflerle oluşturur.

## Mevcut Çıktılar

- K-fold sonuç CSV'si: `results/kfold_results.csv`
- Ortalama/standart sapma özeti: `results/kfold_summary.csv`
- Markdown rapor: `notes/kfold_analysis.md`
- Eğitim çıktıları ve ağırlıklar: `runs/kfold/`

## Tek Chunk Küçük Deneme

Sadece tek chunk/flat SubPipe klasörü varsa, örneğin mevcut Chunk2 kopyası,
tam k-fold protokolünü bozmadan küçük deneme için:

```bash
python3 scripts/convert_pbm_to_png.py
python3 scripts/prepare_single_chunk.py
python3 scripts/run_single_chunk.py --mod LF --epochs 10
python3 scripts/snn_probe.py --mod LF
```

Gerekli SNN bağımlılıkları:

```bash
pip install -r requirements-snn.txt
```

`snn_probe.py`, YOLO checkpoint'ini SpikingJelly ANN-to-SNN dönüşümü açısından
yoklar. Doğrudan conversion denemek için `--convert` eklenebilir.

## Taşınabilir Repo Düzeni

Bu repo başka bir PC'ye taşınırken yalnızca kod, requirements, notlar ve küçük
sonuç CSV/raporları pushlanmalıdır. Aşağıdakiler lokal veya yeniden üretilebilir
çıktı kabul edilir ve `.gitignore` ile dışarıda tutulur:

- `datasets/SubPipe/`: ham SubPipe dataset kopyası
- `datasets/subpipe_kfold/`: `prepare_kfold.py` ile yeniden üretilen YOLO fold datasetleri
- `datasets/subpipe_single_chunk/`: `prepare_single_chunk.py` ile yeniden üretilen küçük dataset
- `runs/`: eğitim klasörleri, checkpointler ve validasyon çıktıları
- `*.pt`, `*.onnx`, `*.engine`: model ağırlıkları ve export dosyaları

Yeni bir makinede devam etmek için:

```bash
git clone <repo-url>
cd tez
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-snn.txt
```

Dataset repo dışındaysa:

```bash
export SUBPIPE_ROOT=/path/to/SubPipe
```

Sonra mevcut veri durumuna göre:

```bash
python3 scripts/convert_pbm_to_png.py
python3 scripts/prepare_kfold.py
```

veya tek chunk pilot çalışma için:

```bash
python3 scripts/convert_pbm_to_png.py
python3 scripts/prepare_single_chunk.py
```

Not: Mevcut git geçmişinde daha önce dataset dosyaları izlenmişse `.gitignore`
tek başına geçmişi küçültmez. Push öncesi ya temiz bir repo kopyası açmak ya da
git geçmişinden dataset dosyalarını ayrıca temizlemek gerekir.

Yeni PC'de Codex ile konuşmaya devam etmek için önce
`notes/codex_handoff.md` ve `notes/snn_probe.md` dosyalarını okutun. İlki kısa
bağlam özeti, ikincisi ayrıntılı deney günlüğüdür.
