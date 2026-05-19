#!/usr/bin/env python3
"""

  1. lid.176.bin  — fastText 176-keelne LID-mudel. Salvestatud asukohta
                    models/lid.176.bin. Vajalik: fasttext, masklid,
                    masklid-cs, fast-langdetect. ~125 MB.
  2. HF-mudelid, mis salvestatakse vahemällu ~/.cache/huggingface (suurus sulgudes):
       - igorsterner/AnE-LID            (~500 MB, transformer)
       - cis-lmu/glotlid                (~1 MB,   üksik .bin)
       - NbAiLab/nb-nordic-lid          (~1 MB,   üksik .ftz)

Kasutamine:
    python scripts/download_models.py            # laadi alla see, mis puudu
    python scripts/download_models.py --force    # lae kõik uuesti alla
    python scripts/download_models.py --skip-hf  # ainult fastTexti fail
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
LID_176_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_176_PATH = MODELS_DIR / "lid.176.bin"
LID_176_LEGACY = PROJECT_ROOT / "lid.176.bin"  # aktsepteeritav, kui pärit vanast paigutusest

HF_SINGLE_FILE_MODELS = [
    ("cis-lmu/glotlid", "model.bin"),
    ("NbAiLab/nb-nordic-lid", "nb-nordic-lid.ftz"),
]
HF_TRANSFORMER_MODELS = [
    "igorsterner/AnE-LID",
]


def download_lid176(force: bool) -> None:
    if not force and (LID_176_PATH.exists() or LID_176_LEGACY.exists()):
        present = LID_176_PATH if LID_176_PATH.exists() else LID_176_LEGACY
        print(f"  ✓ lid.176.bin already present at {present.relative_to(PROJECT_ROOT)}")
        return

    import requests
    from tqdm import tqdm

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  → downloading {LID_176_URL}")
    with requests.get(LID_176_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        tmp = LID_176_PATH.with_suffix(".bin.part")
        with open(tmp, "wb") as fp, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="    lid.176.bin"
        ) as bar:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    fp.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(LID_176_PATH)
    print(f"  ✓ saved to {LID_176_PATH.relative_to(PROJECT_ROOT)}")


def download_hf_single_files(force: bool) -> None:
    from huggingface_hub import hf_hub_download

    for repo_id, filename in HF_SINGLE_FILE_MODELS:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, force_download=force)
            print(f"  ✓ {repo_id}:{filename} cached at {path}")
        except Exception as exc:
            print(f"  ✗ {repo_id}:{filename} — {exc}")


def download_hf_transformers(force: bool) -> None:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    for model_id in HF_TRANSFORMER_MODELS:
        try:
            AutoTokenizer.from_pretrained(model_id, force_download=force)
            AutoModelForTokenClassification.from_pretrained(model_id, force_download=force)
            print(f"  ✓ {model_id} cached")
        except Exception as exc:
            print(f"  ✗ {model_id} — {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--force", action="store_true", help="re-download everything")
    parser.add_argument("--skip-hf", action="store_true", help="only fetch lid.176.bin")
    args = parser.parse_args()

    print("Downloading language-identification models...\n")
    print("[1/3] fastText lid.176.bin")
    download_lid176(args.force)

    if args.skip_hf:
        print("\nSkipping Hugging Face downloads (--skip-hf).")
        return 0

    print("\n[2/3] Hugging Face single-file models (glotlid, nb-nordic-lid)")
    download_hf_single_files(args.force)

    print("\n[3/3] Hugging Face transformer models (AnE-LID)")
    download_hf_transformers(args.force)

    print("\nDone. Run `python evaluation/environment_check.py` to verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
