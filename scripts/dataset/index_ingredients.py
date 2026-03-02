#!/usr/bin/env python3
"""
Set up ChromaDB and index the top N most common ingredients from cleaned data.
Run after clean_and_chunk.py. Run from project root: python scripts/dataset/index_ingredients.py
"""
import argparse
import sys
from pathlib import Path

# ChromaDB uses Pydantic v1 internally, which is incompatible with Python 3.14+.
# See: https://github.com/chroma-core/chroma/issues/5996
if sys.version_info >= (3, 14):
    print(
        "Error: ChromaDB is incompatible with Python 3.14+. Use Python 3.12 or 3.13.\n"
        "Recreate your conda env: conda env remove -n nutrigraph && conda env create -f environment.yml"
    )
    sys.exit(1)

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
TOP_N_DEFAULT = 1000
COLLECTION_NAME = "nutrigraph_ingredients"


def make_document_text(row: pd.Series) -> str:
    """
    Text used for embedding in the ingredient index.

    We embed ONLY the cleaned ingredient name here so that vector similarity
    is dominated by semantic name similarity (e.g. "chicken breast"), while
    nutrient values are stored separately in metadata.
    """
    return str(row.get("name", "")).strip()


def select_top_ingredients(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select top N ingredients: prefer complete nutrients, then USDA, then by source order."""
    df = df.copy()
    df["_complete"] = (
        df["energy_kcal"].notna() & df["protein_g"].notna()
        & df["carbohydrates_g"].notna() & df["fat_g"].notna()
    )
    df["_order"] = (~df["_complete"]).astype(int)
    df = df.sort_values(["_order", "source"])
    return df.head(top_n).drop(columns=["_complete", "_order"], errors="ignore")


def main():
    parser = argparse.ArgumentParser(description="Index top N ingredients into ChromaDB")
    parser.add_argument("-n", "--top", type=int, default=TOP_N_DEFAULT, help=f"Top N ingredients (default {TOP_N_DEFAULT})")
    parser.add_argument("--persist-dir", type=Path, default=CHROMA_DIR, help="ChromaDB persist directory")
    parser.add_argument("--recreate", action="store_true", help="Delete existing collection and recreate")
    args = parser.parse_args()

    parquet_path = PROCESSED_DIR / "ingredients_cleaned.parquet"
    if not parquet_path.exists():
        print("Run clean_and_chunk.py first to create ingredients_cleaned.parquet")
        return

    df = pd.read_parquet(parquet_path)
    df = select_top_ingredients(df, args.top)
    if df.empty:
        print("No ingredients to index.")
        return

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.persist_dir))

    if args.recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted existing collection.")
        except Exception:
            pass

    print("Loading embedding model (sentence-transformers)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_fn(texts):
        return model.encode(texts, show_progress_bar=True).tolist()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "NutriGraph ingredients for RAG retrieval"},
    )

    ids = [f"ing_{i}" for i in range(len(df))]
    documents = [make_document_text(row) for _, row in df.iterrows()]
    metadatas = []
    for _, row in df.iterrows():
        name_str = str(row.get("name", ""))[:500]
        m = {
            "name": name_str,
            # exact_name is the canonical lookup key used by HybridNutritionRetriever
            # for keyword-substring boosting and human-readable result display.
            "exact_name": name_str,
            "source": str(row.get("source", "")),
        }
        for k in ("energy_kcal", "protein_g", "carbohydrates_g", "fat_g"):
            v = row.get(k)
            if v is not None and pd.notna(v):
                try:
                    m[k] = float(v)
                except (TypeError, ValueError):
                    pass
        if row.get("fdc_id") is not None and pd.notna(row.get("fdc_id")):
            m["fdc_id"] = int(row["fdc_id"])
        # brand is populated for OpenFoodFacts rows; absent for USDA rows.
        # Stored in ChromaDB so HybridNutritionRetriever can use
        # where={"brand": {"$eq": brand}} to pre-filter by brand before
        # vector ranking when the LangGraph agent resolves a brand name.
        brand_val = row.get("brand")
        if brand_val is not None and pd.notna(brand_val) and str(brand_val).strip():
            m["brand"] = str(brand_val).strip()[:200]
        metadatas.append(m)

    print(f"Adding {len(ids)} documents to ChromaDB...")
    embeddings = embed_fn(documents)
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"Done. Collection '{COLLECTION_NAME}' has {collection.count()} items. Persisted to {args.persist_dir}")


if __name__ == "__main__":
    main()
