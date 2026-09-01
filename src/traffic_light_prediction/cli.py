"""Command-line entry points for the three workflow stages."""

from __future__ import annotations

import argparse
import json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LISA YOLO traffic-light workflow")
    parser.add_argument(
        "--config", default=".config/config.toml", help="Path to the workflow TOML config"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("data-prep", help="Download and convert LISA")
    prepare.add_argument(
        "--skip-download", action="store_true", help="Convert an already downloaded dataset"
    )
    prepare.add_argument(
        "--force-download", action="store_true", help="Redownload the Kaggle dataset"
    )
    subparsers.add_parser("train", help="Train the configured YOLO11 model")
    evaluate = subparsers.add_parser("evaluate", help="Evaluate and run inference")
    evaluate.add_argument("--source", help="Optional image, directory, or video input")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "data-prep":
        from .data import prepare_dataset

        result = prepare_dataset(
            args.config,
            download=not args.skip_download,
            force_download=args.force_download,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        from .training import train_model

        print(f"Best checkpoint: {train_model(args.config)}")
    else:
        from .evaluation import evaluate_and_infer

        print(json.dumps(evaluate_and_infer(args.config, source=args.source), indent=2))


if __name__ == "__main__":
    main()
