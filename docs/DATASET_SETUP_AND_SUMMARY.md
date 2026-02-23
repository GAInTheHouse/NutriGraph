# Dataset setup and summary

One-time setup: download datasets, clean/chunk, then index the top 1000 ingredients in ChromaDB. Below: how to run it, what you get, and what the data looks like.

---

## Setup

### Prerequisites

- Python 3.10+
- From project root: `pip install -r requirements.txt`

### Steps

1. **Download datasets**  
   From project root:
   ```bash
   python scripts/dataset/download_datasets.py
   ```
   - USDA: Foundation Foods + SR Legacy (JSON) → `data/raw/foundation_food/`, `data/raw/sr_legacy/`
   - OpenFoodFacts: English products CSV (gzip) → `data/raw/en.openfoodfacts.org.products.csv.gz`  
   OpenFoodFacts is large; the script streams the download.

2. **Clean and chunk**  
   ```bash
   python scripts/dataset/clean_and_chunk.py
   ```
   Produces `data/processed/ingredients_cleaned.parquet` and `.csv`.

3. **Index top 1000 in ChromaDB**  
   ```bash
   python scripts/dataset/index_ingredients.py
   ```
   Uses sentence-transformers `all-MiniLM-L6-v2` for embeddings. ChromaDB is stored under `data/chroma/`.  
   Options:
   - `-n 2000` — index top 2000 instead of 1000  
   - `--recreate` — delete existing collection and re-index

### Outputs

| Path | Description |
|------|-------------|
| `data/raw/` | Downloaded USDA zips (extracted) and OpenFoodFacts CSV.gz |
| `data/processed/ingredients_cleaned.parquet` | Cleaned, deduplicated ingredients (name, nutrients, source) |
| `data/chroma/` | ChromaDB persistence; collection `nutrigraph_ingredients` |

The retrieval endpoint can load this ChromaDB and query by ingredient text (semantic search).

---

## Cleaned output: `ingredients_cleaned.csv`

**What it is:** A single table of ingredients from USDA (Foundation Foods, SR Legacy) and OpenFoodFacts, with normalized names and core nutrients. Used for RAG retrieval and for building the ChromaDB index.

| Metric | Value |
|--------|--------|
| **Total rows** | ~71,000 (after deduplication) |
| **Columns** | `source`, `fdc_id`, `name`, `energy_kcal`, `protein_g`, `carbohydrates_g`, `fat_g`, `name_normalized` |
| **Sources** | `usda_foundation`, `usda_sr_legacy`, `openfoodfacts` |

**Example rows (USDA Foundation):**

| name | energy_kcal | protein_g | carbohydrates_g | fat_g |
|------|-------------|-----------|-----------------|-------|
| Hummus, commercial | 229 | 7.35 | 14.9 | 17.1 |
| Beans, black, canned, sodium added, drained and rinsed | — | 6.91 | 19.8 | 1.27 |
| Chicken, breast, boneless, skinless, raw | — | 22.5 | 0 | 1.93 |

**Example rows (OpenFoodFacts):**

| name | energy_kcal | protein_g | carbohydrates_g | fat_g |
|------|-------------|-----------|-----------------|-------|
| Harmons Homestyle Flour Tortillas | 294.2 | 11.76 | 49.02 | 6.86 |
| Corn tortillas | 219.9 | 6.0 | 42.0 | 2.8 |

Nutrients are per 100 g where applicable. Empty cells indicate missing values in the source. Deduplication keeps one row per normalized name, preferring USDA over OpenFoodFacts.

---

## Raw data structure

### 1. Foundation Foods (`data/raw/foundation_food/`)

- **Source:** [USDA FoodData Central](https://fdc.nal.usda.gov/) — Foundation Foods.
- **What it is:** Analytically derived data for minimally processed foods (commodity and commodity-derived), with variability and metadata.
- **Format:** One JSON file (e.g. `FoodData_Central_foundation_food_json_2025-12-18.json`). Root key: `"FoundationFoods"`; value is an array of food objects.
- **Each food has:** `description`, `fdcId`, `foodNutrients` (array of `{ nutrient: { id, name, unitName }, amount }`). We use nutrient IDs 1008 (energy kcal), 1003 (protein), 1005 (carbohydrates), 1004 (fat).
- **Size:** Small (e.g. ~6 MB unzipped); hundreds of foods.

### 2. SR Legacy (`data/raw/sr_legacy/`)

- **Source:** USDA FoodData Central — SR (Standard Reference) Legacy.
- **What it is:** Final release of the legacy Standard Reference database: lab/calculated nutrients and literature-based values. Includes both generic ingredients and branded products.
- **Format:** One JSON file (e.g. `FoodData_Central_sr_legacy_food_json_2018-04.json`). Root key: `"SRLegacyFoods"`; value is an array of food objects.
- **Structure:** Same as Foundation Foods: `description`, `fdcId`, `foodNutrients`, plus optional `foodCategory`, `foodPortions`, etc.
- **Size:** ~205 MB unzipped; thousands of foods.

### 3. OpenFoodFacts (`data/raw/en.openfoodfacts.org.products.csv.gz`)

- **Source:** [Open Food Facts](https://world.openfoodfacts.org/) — English product export.
- **What it is:** Crowdsourced, brand-level packaged foods (barcode, product name, nutrition facts, etc.).
- **Format:** Gzipped CSV (tab-separated). Many columns; we use `product_name`, `energy_100g`, `proteins_100g`, `carbohydrates_100g`, `fat_100g`. Values are per 100 g; energy is in kJ (we convert to kcal when > 500).
- **Size:** Large (hundreds of MB compressed). The clean script reads in chunks and keeps up to 100k rows with at least some nutrient data.

---

## Pipeline in one line

**Download → clean/chunk → index:**  
`download_datasets.py` → `clean_and_chunk.py` → `index_ingredients.py` (all under `scripts/dataset/`). Final cleaned table: `data/processed/ingredients_cleaned.csv` / `.parquet`; vector index: `data/chroma/` (ChromaDB, top 1000 ingredients by default).
