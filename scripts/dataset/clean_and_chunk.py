#!/usr/bin/env python3
"""
Clean and chunk USDA + OpenFoodFacts data into a unified ingredients table.
Output: data/processed/ingredients_cleaned.parquet (and .csv for inspection).
Run after download_datasets.py. Run from project root: python scripts/dataset/clean_and_chunk.py
"""
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# USDA nutrient IDs (FoodData Central)
NUTRIENT_IDS = {
    1008: "energy_kcal",
    1003: "protein_g",
    1005: "carbohydrates_g",
    1004: "fat_g",
}


def extract_nutrients(food: dict) -> dict:
    """Extract energy, protein, carbs, fat from USDA foodNutrients."""
    out = {v: None for v in NUTRIENT_IDS.values()}
    for fn in food.get("foodNutrients") or []:
        nut = fn.get("nutrient") or {}
        nid = nut.get("id")
        if nid in NUTRIENT_IDS:
            out[NUTRIENT_IDS[nid]] = fn.get("amount")
    return out


def load_usda_foundation(raw_dir: Path) -> list[dict]:
    """Load Foundation Foods JSON (single file or folder of JSON)."""
    base = raw_dir / "foundation_food"
    if not base.exists():
        return []
    rows = []
    for path in base.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        foods = (data.get("FoundationFoods") or data.get("foundationFoods")) if isinstance(data, dict) else data
        if foods is None or not isinstance(foods, list):
            foods = data if isinstance(data, list) else []
        if not foods:
            continue
        for f in foods:
            name = (f.get("description") or "").strip()
            if not name:
                continue
            nut = extract_nutrients(f)
            rows.append({
                "source": "usda_foundation",
                "fdc_id": f.get("fdcId"),
                "name": name,
                **nut,
            })
    return rows


def load_usda_sr_legacy(raw_dir: Path) -> list[dict]:
    """Load SR Legacy JSON."""
    base = raw_dir / "sr_legacy"
    if not base.exists():
        return []
    rows = []
    for path in base.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Official key in download is SRLegacyFoods
        foods = (
            (
                data.get("SRLegacyFoods")
                or data.get("SR Legacy Food")
                or data.get("srLegacyFood")
            )
            if isinstance(data, dict)
            else data
        )
        if foods is None or not isinstance(foods, list):
            foods = data if isinstance(data, list) else []
        if not foods:
            continue
        for f in foods:
            name = (f.get("description") or "").strip()
            if not name:
                continue
            nut = extract_nutrients(f)
            rows.append({
                "source": "usda_sr_legacy",
                "fdc_id": f.get("fdcId"),
                "name": name,
                **nut,
            })
    return rows


def normalize_name(s: str) -> str:
    """Lowercase, collapse whitespace."""
    if not s or not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s.lower().strip())
    return s


def load_openfoodfacts(raw_dir: Path, max_rows: int = 100_000) -> list[dict]:
    """Load OpenFoodFacts CSV in chunks; keep rows with key nutrients."""
    gz_path = raw_dir / "en.openfoodfacts.org.products.csv.gz"
    if not gz_path.exists():
        return []
    rows = []
    cols = ["product_name", "energy_100g", "proteins_100g", "carbohydrates_100g", "fat_100g"]
    try:
        for chunk in pd.read_csv(gz_path, compression="gzip", sep="\t", usecols=cols, dtype=str, chunksize=50000):
            chunk = chunk.dropna(subset=["product_name"])
            chunk = chunk.dropna(subset=["energy_100g", "proteins_100g", "carbohydrates_100g", "fat_100g"], how="all")
            for _, r in chunk.iterrows():
                name = (r.get("product_name") or "").strip()
                if not name or len(name) < 2:
                    continue

                def _parse(val):
                    if val is None or (isinstance(val, str) and val.strip() in ("", "nan")):
                        return None
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None

                energy = _parse(r.get("energy_100g"))
                protein = _parse(r.get("proteins_100g"))
                carb = _parse(r.get("carbohydrates_100g"))
                fat = _parse(r.get("fat_100g"))
                if energy is not None and energy > 500:
                    energy = energy / 4.184
                rows.append({
                    "source": "openfoodfacts",
                    "fdc_id": None,
                    "name": name,
                    "energy_kcal": energy,
                    "protein_g": protein,
                    "carbohydrates_g": carb,
                    "fat_g": fat,
                })
                if len(rows) >= max_rows:
                    return rows
    except Exception as e:
        print(f"OpenFoodFacts read warning: {e}")
    return rows


def clean_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize names, dedupe by normalized name (prefer USDA then OFF)."""
    df = df[df["name"].astype(str).str.len() >= 2].copy()
    df["name_normalized"] = df["name"].apply(normalize_name)
    df = df[df["name_normalized"].str.len() >= 2]
    df["_order"] = df["source"].map({"usda_foundation": 0, "usda_sr_legacy": 1, "openfoodfacts": 2})
    df = df.sort_values("_order").drop_duplicates(subset=["name_normalized"], keep="first").drop(columns=["_order"])
    return df


def main():
    print("Loading USDA Foundation Foods...")
    foundation = load_usda_foundation(RAW_DIR)
    print("Loading USDA SR Legacy...")
    sr = load_usda_sr_legacy(RAW_DIR)
    print("Loading OpenFoodFacts (chunked, up to 100k rows with nutrients)...")
    off = load_openfoodfacts(RAW_DIR)

    df = pd.DataFrame(foundation + sr + off)
    if df.empty:
        print("No data loaded. Run download_datasets.py first.")
        return

    print("Cleaning and deduplicating...")
    df = clean_and_dedupe(df)

    out_parquet = OUT_DIR / "ingredients_cleaned.parquet"
    out_csv = OUT_DIR / "ingredients_cleaned.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows -> {out_parquet} and {out_csv}")


if __name__ == "__main__":
    main()
