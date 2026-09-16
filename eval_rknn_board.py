#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from rknn_eval.board.metrics import PerplexityAccumulator
from rknn_eval.board.ppl_runner import RKNNModelFiles, RKNNPerplexityRunner
from rknn_eval.data.schema import validate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-tokenized perplexity records on an RK1828 with RKNN3Lite."
    )
    parser.add_argument("--model", type=Path, required=True, help="RKNN model file")
    parser.add_argument("--weight", type=Path, required=True, help="RKNN weight file")
    parser.add_argument("--embed", type=Path, required=True, help="FP16 embedding table")
    parser.add_argument("--data", type=Path, required=True, help="prepared JSONL dataset")
    parser.add_argument("--output", type=Path, required=True, help="per-record JSONL output")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--target", default="rk1828")
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--core-mask", type=lambda value: int(value, 0), default=0xFF)
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--logits-name", default="logits")
    parser.add_argument("--special-bos-id", type=int, default=11)
    parser.add_argument("--special-eos-id", type=int, default=248046)
    parser.add_argument("--linefeed-id", type=int, default=198)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_record(record)
            if record["type"] != "perplexity":
                raise ValueError(f"{path}:{line_number}: expected a perplexity record")
            yield record


def load_existing(path: Path) -> tuple[set[str], PerplexityAccumulator]:
    completed: set[str] = set()
    total = PerplexityAccumulator()
    if not path.exists():
        return completed, total
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                record_id = str(item["id"])
                nll_sum = float(item["nll_sum"])
                scored_tokens = int(item["scored_tokens"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid resume output {path}:{line_number}") from exc
            if record_id in completed:
                raise ValueError(f"duplicate record id in resume output: {record_id}")
            completed.add(record_id)
            total.add(nll_sum, scored_tokens)
    return completed, total


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.skip < 0:
        raise ValueError("--skip must be non-negative")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed, total = (
        (set(), PerplexityAccumulator())
        if args.no_resume
        else load_existing(args.output)
    )
    mode = "w" if args.no_resume else "a"
    selected = 0
    processed = 0
    started = time.monotonic()

    files = RKNNModelFiles(args.model, args.weight, args.embed)
    with RKNNPerplexityRunner(
        files,
        target=args.target,
        core_mask=args.core_mask,
        max_context_len=args.max_context_len,
        device_id=args.device_id,
        logits_name=args.logits_name,
        special_bos_id=args.special_bos_id,
        special_eos_id=args.special_eos_id,
        linefeed_id=args.linefeed_id,
        verbose=args.verbose,
    ) as runner, args.output.open(mode, encoding="utf-8", buffering=1) as output:
        print(
            json.dumps(
                {
                    "event": "runtime_ready",
                    "model_config": runner.model_config,
                    "sdk_version": runner.sdk_version,
                    "core_mask": hex(args.core_mask),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for index, record in enumerate(iter_records(args.data)):
            if index < args.skip or record["id"] in completed:
                continue
            if args.limit is not None and selected >= args.limit:
                break
            selected += 1
            score = runner.score(record["tokens"], record["score_from"])
            total.add(score.nll_sum, score.scored_tokens)
            processed += 1
            item = {
                "schema_version": 1,
                "id": record["id"],
                "task": record["task"],
                "model": args.model_name or args.model.stem,
                "nll_sum": score.nll_sum,
                "scored_tokens": score.scored_tokens,
                "mean_nll": score.mean_nll,
                "perplexity": math.exp(score.mean_nll),
                "elapsed_seconds": score.elapsed_seconds,
                "n_prefill_tokens": score.n_prefill_tokens,
                "n_decode_tokens": score.n_decode_tokens,
            }
            output.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(json.dumps({"event": "record_done", **item}, ensure_ascii=False), flush=True)

    summary = {
        "schema_version": 1,
        "task": "perplexity",
        "model": args.model_name or args.model.stem,
        "data": str(args.data),
        "records_processed_this_run": processed,
        "records_total_in_output": len(completed) + processed,
        "nll_sum": total.nll_sum,
        "scored_tokens": total.scored_tokens,
        "mean_nll": total.mean_nll,
        "perplexity": total.perplexity,
        "elapsed_seconds": time.monotonic() - started,
        "host": platform.node(),
        "python": sys.version.split()[0],
        "core_mask": hex(args.core_mask),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, summary_path)
    print(json.dumps({"event": "summary", **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
