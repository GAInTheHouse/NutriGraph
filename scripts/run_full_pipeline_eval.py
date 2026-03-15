#!/usr/bin/env python3
"""
CLI entry point: Full Pipeline (Image → Macros) evaluation.

Uploads each dish image to POST /api/v1/analyze-dish and compares the
returned macro totals (calories, protein, carbs, fat, fiber) against
the ground-truth values in the golden set.

Metrics computed
----------------
  • MAE per macro (calories, protein, carbs, fat, fiber)
  • MAPE for calories
  • Coverage (fraction of images successfully analysed)
  • Latency percentiles (mean, P50, P90, P99)
  • LLM-as-a-judge quality score (default ON)

Artifacts written
-----------------
  artifacts/full_pipeline_records.csv
  artifacts/full_pipeline_metrics.json
  artifacts/full_pipeline_report.md

Usage
-----
  python scripts/run_full_pipeline_eval.py \\
      --images-dir data/images \\
      --golden-csv data/golden_set.csv \\
      --api-base-url http://localhost:8000 \\
      [--no-judge] \\
      [--timeout 120] \\
      [--artifacts-dir artifacts] \\
      [--log-level INFO]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.image_eval.client import ImageEvalClient
from src.eval.image_eval.report import (
    generate_full_pipeline_report,
    print_full_pipeline_summary,
)
from src.eval.image_eval.runner import run_full_pipeline_eval


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NutriGraph full pipeline (image → macros) evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/images"),
        help="Directory containing images named 1.jpg … 200.jpg",
    )
    p.add_argument(
        "--golden-csv",
        type=Path,
        default=Path("data/golden_set.csv"),
        help="Path to the golden set CSV (200 rows)",
    )
    p.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="NutriGraph backend base URL",
    )
    p.add_argument(
        "--judge-base-url",
        default=None,
        help="Judge endpoint base URL (defaults to --api-base-url)",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for output CSVs, JSON metrics, and Markdown report",
    )
    p.add_argument(
        "--no-judge",
        dest="run_judge",
        action="store_false",
        default=True,
        help="Disable the LLM-as-a-judge call (faster, cheaper)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout per request (seconds)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    judge_url = args.judge_base_url or args.api_base_url

    if not args.golden_csv.exists():
        print(f"ERROR: golden CSV not found: {args.golden_csv}", file=sys.stderr)
        sys.exit(1)

    if not args.images_dir.exists():
        print(
            f"WARNING: images directory not found: {args.images_dir}\n"
            "         All dishes will be recorded as missing-image failures.",
            file=sys.stderr,
        )

    client = ImageEvalClient(
        api_base_url=args.api_base_url,
        judge_base_url=judge_url,
        timeout=args.timeout,
    )

    records, metrics = run_full_pipeline_eval(
        golden_csv=args.golden_csv,
        images_dir=args.images_dir,
        client=client,
        artifacts_dir=args.artifacts_dir,
        run_judge=args.run_judge,
    )

    print_full_pipeline_summary(metrics)

    report_path = args.artifacts_dir / "full_pipeline_report.md"
    generate_full_pipeline_report(records, metrics, report_path)
    print(f"Markdown report → {report_path}")


if __name__ == "__main__":
    main()
