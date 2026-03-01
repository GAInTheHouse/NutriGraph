#!/usr/bin/env python3
"""
Quick CLI to test the LangGraph clarification state machine.

Run from project root:

    python scripts/test_clarification.py

It will:
  - Invoke the clarification graph on a sample list of ingredients.
  - Print the threshold and per-ingredient scores.
  - Indicate which ingredients are low-confidence.
  - Show any generated clarification questions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import sys

# Ensure project root (containing `src/`) is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.clarification_graph import build_clarification_graph


def main() -> None:
    # You can edit this list to try different examples.
    ingredients: List[str] = [
        "chicken breast",
        "mystery sauce",
        "side salad",
    ]
    threshold = 0.7

    graph = build_clarification_graph(default_threshold=threshold)

    state_in = {
        "ingredients": ingredients,
        "threshold": threshold,
    }

    result = graph.invoke(state_in)

    scores = result.get("scores", {})
    matches_by_ing = result.get("matches", {})
    low_conf = set(result.get("low_conf_ingredients", []))
    thr = result.get("threshold", threshold)

    print("=== Clarification Graph Test ===")
    print(f"Threshold: {thr:.2f}")
    print()
    print("Per-ingredient scores and top match (including nutrients):")
    for ing in ingredients:
        s = scores.get(ing)
        mlist = matches_by_ing.get(ing, []) or []

        if s is None or not mlist:
            print(f"  - {ing!r}: score = N/A (no matches)")
            continue

        status = "LOW CONFIDENCE" if ing in low_conf else "ok"
        print(f"  - {ing!r}: score = {s:.3f}  [{status}]")

        top = mlist[0]
        name = top.get("name", "")
        source = top.get("source", "")
        dist = top.get("distance", None)
        kcal = top.get("energy_kcal", None)
        prot = top.get("protein_g", None)
        carb = top.get("carbohydrates_g", None)
        fat = top.get("fat_g", None)

        dist_str = f"{dist:.3f}" if isinstance(dist, (int, float)) else "N/A"
        print(f"      top match: {name!r} (source={source}, distance={dist_str})")

        # Print nutrient info if available
        macro_parts = []
        if isinstance(kcal, (int, float)):
            macro_parts.append(f"{kcal:.1f} kcal")
        if isinstance(prot, (int, float)):
            macro_parts.append(f"{prot:.1f} g protein")
        if isinstance(carb, (int, float)):
            macro_parts.append(f"{carb:.1f} g carbs")
        if isinstance(fat, (int, float)):
            macro_parts.append(f"{fat:.1f} g fat")

        if macro_parts:
            print(f"      nutrients: {', '.join(macro_parts)}")

    print()
    questions = result.get("questions", [])
    if questions:
        print("Clarification questions:")
        for q in questions:
            print(f"  - {q}")
    else:
        print("No clarification questions generated (all ingredients above threshold).")


if __name__ == "__main__":
    main()

