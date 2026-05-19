from __future__ import annotations

import importlib
from pathlib import Path


MODULES = [
    ("lingua", "lingua-language-detector"),
    ("langid", "langid"),
    ("fasttext", "fasttext-predict"),
    ("fast_langdetect", "fast-langdetect"),
    ("huggingface_hub", "huggingface-hub"),
    ("transformers", "transformers"),
    ("heliport", "heliport"),
    ("pycld2", "pycld2"),
]

FILES = [
    Path("lid.176.bin"),
]


def check() -> None:
    print("Environment check:\n")
    print("Python modules:")
    for module_name, pip_name in MODULES:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {module_name}")
        except Exception:
            print(f"  ✗ {module_name} (pip install {pip_name})")

    print("\nLocal model files:")
    for path in FILES:
        if path.exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} (missing)")

    print("\nRecommended setup order:")
    print("  1) pip install -r requirements.txt && pip install -e .")
    print("  2) python scripts/download_models.py")
    print("  3) python scripts/run_pipeline.py --help")


if __name__ == "__main__":
    check()
