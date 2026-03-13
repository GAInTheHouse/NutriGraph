"""
CLI entry point for the NutriGraph offline evaluation framework.

Usage
-----
python scripts/run_eval.py \\
    --golden-csv data/golden_set.csv \\
    --api-base-url http://localhost:8000 \\
    --judge-base-url http://localhost:8000 \\
    --repeats 1 \\
    --timeout 60 \\
    --max-turns 5 \\
    --artifacts-dir artifacts \\
    --gemini-model gemini-1.5-flash

Environment variables
---------------------
VERTEXAI_API_KEY : Google AI API key for the LLM auto-responder (required).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure the project root is on sys.path when the script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.auto_responder import AutoResponder
from src.eval.client import NutriGraphEvalClient
from src.eval.report import generate_markdown_report, print_summary
from src.eval.runner import run_evaluation


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="NutriGraph offline evaluation framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--golden-csv",
        default="data/golden_set.csv",
        help="Path to the golden set CSV file.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("NUTRIGRAPH_BACKEND_URL", "http://localhost:8000"),
        help="Base URL of the NutriGraph analysis backend.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("NUTRIGRAPH_BACKEND_URL", "http://localhost:8000"),
        help="Base URL of the NutriGraph judge backend (may be the same).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to evaluate each dish (>1 enables consistency analysis).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum clarification rounds per dish before raising an error.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory where eval_records.csv, summary_metrics.json, and eval_report.md are saved.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-1.5-flash",
        help="Gemini model name for the LLM auto-responder.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    load_dotenv()

    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_eval")

    # -----------------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------------
    golden_csv = Path(args.golden_csv)
    if not golden_csv.exists():
        logger.error("Golden set CSV not found: %s", golden_csv)
        sys.exit(1)

    api_key = os.environ.get("VERTEXAI_API_KEY")
    if not api_key:
        logger.error(
            "VERTEXAI_API_KEY environment variable is not set.  "
            "The LLM auto-responder cannot be initialised."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Instantiate components
    # -----------------------------------------------------------------------
    logger.info("Initialising AutoResponder (model=%s) …", args.gemini_model)
    auto_responder = AutoResponder(api_key=api_key, model=args.gemini_model)

    logger.info(
        "Initialising NutriGraphEvalClient (api=%s, judge=%s, timeout=%ds, max_turns=%d) …",
        args.api_base_url,
        args.judge_base_url,
        args.timeout,
        args.max_turns,
    )
    client = NutriGraphEvalClient(
        api_base_url=args.api_base_url,
        judge_base_url=args.judge_base_url,
        auto_responder=auto_responder,
        timeout=args.timeout,
        max_turns=args.max_turns,
    )

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    logger.info(
        "Starting evaluation: golden_csv=%s, repeats=%d, artifacts_dir=%s",
        golden_csv,
        args.repeats,
        args.artifacts_dir,
    )

    _records, summary = run_evaluation(
        golden_csv=golden_csv,
        client=client,
        repeats=args.repeats,
        artifacts_dir=args.artifacts_dir,
    )

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    print_summary(summary)

    report_path = generate_markdown_report(
        summary,
        output_path=Path(args.artifacts_dir) / "eval_report.md",
    )
    logger.info("Markdown report saved → %s", report_path)

    # Quick console overview
    cov = summary.get("coverage", 0.0)
    mae_cal = summary.get("mae", {}).get("calories_kcal", float("nan"))
    mean_q = summary.get("agent_efficiency", {}).get("mean_questions", float("nan"))
    judge_mean = summary.get("judge_stats", {}).get("mean", float("nan"))
    lat_p95 = summary.get("latency_stats", {}).get("p95", float("nan"))

    print("Quick overview:")
    print(f"  Coverage            : {cov:.1%}")
    print(f"  MAE calories        : {mae_cal:.1f} kcal")
    print(f"  Mean questions      : {mean_q:.2f}")
    print(f"  Judge mean score    : {judge_mean:.2f} / 10")
    print(f"  Latency p95         : {lat_p95:.1f} s")
    print(f"\nArtifacts saved to   : {args.artifacts_dir}/")


if __name__ == "__main__":
    main()
