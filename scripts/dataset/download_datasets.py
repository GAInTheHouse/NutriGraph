#!/usr/bin/env python3
"""
Download USDA FoodData Central and OpenFoodFacts datasets for NutriGraph.
Saves to data/raw/. Run from project root: python scripts/dataset/download_datasets.py
"""
import zipfile
import requests
from pathlib import Path
from urllib.parse import urlparse

# Project root: parent of scripts/dataset/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# USDA: Foundation Foods + SR Legacy (good for ingredients; manageable size)
USDA_URLS = {
    "foundation_food": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_json_2025-12-18.zip",
    "sr_legacy": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_json_2018-04.zip",
}

# OpenFoodFacts: English products CSV (gzipped). Large file; streamed.
OPENFOODFACTS_CSV_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"


def download_file(url: str, dest: Path, stream: bool = False) -> None:
    """Download url to dest. If stream=True, use streaming for large files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "NutriGraph-DataPipeline/1.0"}
    r = requests.get(url, headers=headers, stream=stream, timeout=60)
    r.raise_for_status()
    if stream:
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        dest.write_bytes(r.content)
    print(f"  -> {dest}")


def unzip(zip_path: Path, out_dir: Path) -> None:
    """Unzip into out_dir."""
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print(f"  extracted to {out_dir}")


def main():
    print("NutriGraph — downloading datasets to data/raw/\n")

    # --- USDA ---
    print("1. USDA FoodData Central (Foundation Foods + SR Legacy)")
    for name, url in USDA_URLS.items():
        zip_name = Path(urlparse(url).path).name
        zip_path = RAW_DIR / zip_name
        if zip_path.exists():
            print(f"  Skip (exists): {zip_name}")
        else:
            print(f"  Downloading {name}...")
            download_file(url, zip_path)
        out_dir = RAW_DIR / name
        if not out_dir.exists() or not any(out_dir.iterdir()):
            if zip_path.exists():
                unzip(zip_path, out_dir)

    # --- OpenFoodFacts ---
    print("\n2. OpenFoodFacts (en products CSV)")
    off_gz = RAW_DIR / "en.openfoodfacts.org.products.csv.gz"
    if off_gz.exists():
        print(f"  Skip (exists): {off_gz.name}")
    else:
        print("  Downloading (streaming; may take a while)...")
        download_file(OPENFOODFACTS_CSV_URL, off_gz, stream=True)

    print("\nDone. Raw data under data/raw/")


if __name__ == "__main__":
    main()
