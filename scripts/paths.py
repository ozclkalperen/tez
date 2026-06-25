from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
KFOLD_DIR = DATASETS_DIR / "subpipe_kfold"
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"
NOTES_DIR = REPO_ROOT / "notes"


def _looks_like_full_subpipe(path: Path) -> bool:
    return all((path / f"Chunk{i}").exists() for i in range(5))


def _looks_like_flat_subpipe(path: Path) -> bool:
    return all(
        (path / name / "Image").exists() and (path / name / "YOLO_Annotation").exists()
        for name in ["SSS_LF_images", "SSS_HF_images"]
    )


def get_subpipe_root(require_full: bool = True) -> Path:
    env_path = os.environ.get("SUBPIPE_ROOT")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            DATASETS_DIR / "SubPipe",
            DATASETS_DIR / "subpipe",
            REPO_ROOT / "SubPipe",
            REPO_ROOT / "subpipe",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if _looks_like_full_subpipe(candidate):
            return candidate
        if not require_full and _looks_like_flat_subpipe(candidate):
            return candidate

    searched = "\n  - ".join(str(p) for p in candidates)
    expected = "Chunk0..Chunk4 yapısı" if require_full else "Chunk0..Chunk4 veya flat SSS_* yapısı"
    raise FileNotFoundError(
        "SubPipe dataset klasörü bulunamadı.\n"
        "Dataset'i repoda `datasets/SubPipe` veya `datasets/subpipe` altına koyun; "
        "alternatif olarak SUBPIPE_ROOT ortam değişkenini ayarlayın.\n"
        f"Beklenen yapı: {expected}.\n"
        f"Aranan yerler:\n  - {searched}"
    )
