# Codex Handoff

Bu dosya, yeni PC veya yeni Codex oturumunda calismaya kaldigimiz yerden devam
edebilmek icin kisa baglam notudur. Yeni oturumda su cumle yeterli olur:

```text
notes/codex_handoff.md ve notes/snn_probe.md dosyalarini oku, tez SNN
calismasina kaldigimiz yerden devam edelim.
```

## Proje Amaci

Tez konusu: SNN ile su altinda boru tespiti. Calisma SubPipe side-scan sonar
veri setinde tek sinifli `pipe` tespiti uzerine kurulu. ANN baseline YOLOv26n,
SNN tarafi ise simdilik train-then-convert / hybrid-SNN rotasinda ilerliyor.

## Veri Durumu

- Ham SubPipe dataset git'e konmuyor.
- Lokal dataset yolu: `datasets/SubPipe/` veya repo disinda `SUBPIPE_ROOT`.
- Su anki pilot calisma tek chunk/flat veri uzerinden yurudu.
- Full veri geldiginde ayni scriptler k-fold/full protokole tasinacak.
- `datasets/subpipe_kfold/` ve `datasets/subpipe_single_chunk/` yeniden
  uretilebilir cikti kabul ediliyor.

## Repo Tasima Karari

- Git'e girecekler: `scripts/`, `notes/`, kucuk `results/` CSV/TXT dosyalari,
  `README.md`, `requirements-snn.txt`, `.gitignore`.
- Git'e girmeyecekler: `datasets/SubPipe/`, uretilmis YOLO datasetleri,
  `runs/`, checkpoint/export dosyalari.
- `.gitignore` eklendi.
- Mevcut repo gecmisinde `datasets/subpipe_kfold` izlenmis durumda; push oncesi
  index/gecmis temizligi veya temiz repo/orphan branch gerekecek.

## Ana SNN Metodolojisi

Ana rota sifirdan full-SNN egitmek degil:

1. YOLOv26n ANN modelini ReLU aktivasyonla egit.
2. Secili feature extraction bloklarini channel-wise kalibre edilmis IF spiking
   node'lara donustur.
3. Detection head'i analog birak.
4. ANN ve SNN/hybrid-SNN performansini ayni splitlerde karsilastir.

Bu nedenle tezde dil durust tutulacak: model tam full-SNN degil, YOLOv26n
tabanli train-then-convert hybrid-SNN detector.

## Mevcut Ana Aday

Pilot tek-chunk icin en dengeli aday:

- Scope: `range5-10+range13-13+range17-22`
- Calibration: channel-wise `p95 * 0.4`
- Timesteps: `T=16`
- Spike scale: `threshold`
- Cikti klasoru:
  `results/single_chunk/train_then_convert_snn_p95_channel_t16/`

Ana karsilastirma:

| Modality | Split | ANN mAP50-95 | SNN mAP50-95 | Retention |
|----------|-------|--------------|--------------|-----------|
| LF | val | 0.6687 | 0.6032 | 90.22% |
| LF | test | 0.7389 | 0.6517 | 88.20% |
| HF | val | 0.5479 | 0.5423 | 98.97% |
| HF | test | 0.5316 | 0.5148 | 96.83% |

Full/backbone spiking denemesi ayni kalibrasyonla basarisiz oldu:

- LF backbone/all: test mAP50-95 = 0.0000
- HF backbone/all: test mAP50-95 = 0.0000

Bu bulgu, kademeli/secili blok donusumunun neden gerekli oldugunu gosteriyor.

## Onemli Scriptler

- `scripts/paths.py`: repo-relative ve `SUBPIPE_ROOT` destekli path merkezi.
- `scripts/prepare_single_chunk.py`: tek chunk pilot YOLO dataseti.
- `scripts/run_single_chunk.py`: LF/HF ANN baseline egitimi.
- `scripts/run_hybrid_snn.py`: tek checkpoint uzerinde hybrid SNN denemeleri.
- `scripts/run_hybrid_batch.py`: single/kfold uzerinde batch SNN kosulari.
- `scripts/run_train_then_convert_snn.py`: ANN eval + SNN eval ana protokol.
- `scripts/summarize_ann_snn_comparison.py`: ana tablo ozeti.

## Yeni PC'de Baslangic

```bash
git clone <repo-url>
cd tez
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-snn.txt
export SUBPIPE_ROOT=/path/to/SubPipe
python3 scripts/prepare_single_chunk.py
```

Full veri varsa:

```bash
python3 scripts/prepare_kfold.py
python3 scripts/verify_kfold.py
```

## Siradaki Mantikli Is

1. Once repoyu pushlanabilir hale getir: datasetleri index/gecmisten ayir.
2. Sonra pilot tek-chunk protokolunu full veri/k-fold uzerinde ayni ayarlarla
   kos.
3. Ana tez tablosunu ANN vs SNN retention olarak raporla.
4. Ek bolumde full/backbone donusumun neden kirildigini ve secili blok
   stratejisinin nasil secildigini ablation olarak anlat.
