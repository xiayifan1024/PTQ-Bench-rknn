#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rknn_eval.data.prepare import (
    DataPreparationError,
    prepare_dataset,
    validate_prepared_file,
)


def _load_config(path: str) -> dict:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise DataPreparationError(
                "YAML configuration requires PyYAML; install it or use a JSON config"
            ) from exc
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            raise DataPreparationError(f"invalid YAML configuration: {exc}") from exc
    if not isinstance(config, dict):
        raise DataPreparationError("configuration root must be an object")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic JSONL datasets for RKNN board evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="prepare a dataset")
    prepare_parser.add_argument("--config", required=True, help="YAML configuration path")

    validate_parser = subparsers.add_parser("validate", help="validate prepared JSONL")
    validate_parser.add_argument("--input", required=True, help="prepared JSONL path")

    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_dataset(_load_config(args.config))
        else:
            result = validate_prepared_file(args.input)
    except (OSError, json.JSONDecodeError, DataPreparationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
