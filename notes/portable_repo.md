# Taşınabilir Repo Notu

Amaç: Tez kodunu, deney notlarını ve küçük sonuç özetlerini git ile taşımak;
ham datasetleri, üretilmiş YOLO datasetlerini ve eğitim ağırlıklarını git dışında
tutmak.

## Git'e girmesi gerekenler

- `scripts/`: veri hazırlama, eğitim, SNN dönüşüm ve özet scriptleri
- `notes/`: tez günlüğü, literatür/deney notları
- `results/`: küçük CSV/TXT özetleri
- `README.md`, `requirements-snn.txt`, `.gitignore`

## Git'e girmemesi gerekenler

- `datasets/SubPipe/`: ham SubPipe dataset
- `datasets/subpipe_kfold/`: scriptlerle yeniden üretilen k-fold YOLO dataset
- `datasets/subpipe_single_chunk/`: scriptlerle yeniden üretilen pilot dataset
- `runs/`: YOLO eğitim klasörleri, validasyon çıktıları ve checkpointler
- `*.pt`, `*.onnx`, `*.engine`: model ağırlıkları ve export dosyaları

## Başka PC'de devam etme akışı

1. Repoyu klonla.
2. Sanal ortamı kur ve bağımlılıkları yükle.
3. SubPipe datasetini lokal diskte tut.
4. Dataset repo dışındaysa `SUBPIPE_ROOT` ayarla.
5. `prepare_kfold.py` veya `prepare_single_chunk.py` ile YOLO datasetini yeniden üret.
6. Eğitim/SNN scriptlerini çalıştır; yeni çıktılar lokal `runs/` ve `results/` altına düşer.

## Önemli durum

Bu klasörde `.git` boyutu yaklaşık 7.3GB görünüyor ve `datasets/subpipe_kfold`
dosyaları git tarafından izlenmiş durumda. `.gitignore` bundan sonra yeni ağır
dosyaları engeller; fakat geçmişte izlenmiş dosyaları tek başına temizlemez.

Push için iki pratik yol var:

1. En temiz yol: Kod, notlar ve küçük sonuçlarla yeni bir temiz repo oluşturmak.
2. Mevcut repo geçmişini koruyacaksak: `git rm --cached` ile datasetleri index'ten
   çıkarmak ve gerekiyorsa `git filter-repo`/BFG ile geçmişi temizlemek.

Tez çalışması açısından en güvenli yaklaşım, datasetleri ayrı taşımak ve repoyu
yeniden üretilebilir deney reçetesi olarak tutmaktır.

## Codex Sohbet Devamlılığı

Codex sohbet geçmişinin tamamına yeni PC'de otomatik güvenmemek gerekir. Bu
yüzden çalışma bağlamı repoda not olarak tutulur:

- `notes/codex_handoff.md`: yeni oturum için kısa bağlam ve sıradaki adımlar
- `notes/snn_probe.md`: ayrıntılı SNN deney günlüğü ve sonuç yorumları

Yeni PC'de yeni Codex oturumu açınca şu komut yeterli bağlamı verir:

```text
notes/codex_handoff.md ve notes/snn_probe.md dosyalarını oku, tez SNN
çalışmasına kaldığımız yerden devam edelim.
```
