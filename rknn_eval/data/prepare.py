from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .gguf import GGUFError, tokenizer_fingerprint
from .schema import SCHEMA_VERSION, DataValidationError, validate_record


class DataPreparationError(RuntimeError):
    """Raised when source data cannot be converted into the prepared schema."""


def prepare_dataset(config: Mapping[str, Any], tokenizer: Any = None) -> dict[str, Any]:
    """Prepare one dataset and atomically write JSONL plus a manifest.

    ``tokenizer`` is injectable to make the PPL preparation path testable without
    loading Transformers. When omitted, it is loaded from ``config.tokenizer``.
    """

    task = _require_mapping(config, "task")
    source = _require_mapping(config, "source")
    output = _require_mapping(config, "output")
    task_name = _require_string(task, "name")
    task_type = _require_string(task, "type")
    if task_type not in {"generation", "multiple_choice", "perplexity"}:
        raise DataPreparationError(f"unsupported task.type: {task_type!r}")

    source_rows = _load_rows(source)
    perplexity_mode = str(config.get("perplexity_mode", "concatenate"))
    if task_type == "perplexity" and perplexity_mode == "document_windows":
        records, tokenizer_info = _prepare_perplexity(
            source_rows, config, task_name, tokenizer=tokenizer
        )
    else:
        rows = _materialize_rows(source_rows, source, config.get("sampling", {}))
        if not rows:
            raise DataPreparationError("source produced no rows after sampling")
        if task_type == "perplexity":
            records, tokenizer_info = _prepare_perplexity(
                rows, config, task_name, tokenizer=tokenizer
            )
        else:
            records = _prepare_examples(rows, config, task_name, task_type)
            tokenizer_info = None

    _validate_records(records)
    model_binding_info = (
        _model_bindings(config["model_bindings"])
        if config.get("model_bindings") is not None
        else None
    )

    output_path = Path(_require_string(output, "path")).expanduser()
    manifest_path = Path(
        output.get("manifest_path", f"{output_path}.manifest.json")
    ).expanduser()
    if output_path.resolve() == manifest_path.resolve():
        raise DataPreparationError("output.path and output.manifest_path must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    digest = _write_jsonl_atomic(output_path, records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task_name,
        "type": task_type,
        "source": _source_manifest(source),
        "sampling": _json_compatible(config.get("sampling", {})),
        "configuration_sha256": _sha256_json(config),
        "record_count": len(records),
        "output_path": str(output_path),
        "output_sha256": digest,
    }
    if tokenizer_info is not None:
        manifest["tokenizer"] = tokenizer_info
    if model_binding_info is not None:
        manifest["model_bindings"] = model_binding_info
    _write_json_atomic(manifest_path, manifest)
    return manifest


def validate_prepared_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    prepared_path = Path(path)
    digest = hashlib.sha256()
    ids: set[str] = set()
    count = 0
    with prepared_path.open("rb") as raw_file:
        for line_number, raw_line in enumerate(raw_file, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
                validate_record(record)
            except (json.JSONDecodeError, DataValidationError) as exc:
                raise DataPreparationError(
                    f"{prepared_path}:{line_number}: {exc}"
                ) from exc
            record_id = record["id"]
            if record_id in ids:
                raise DataPreparationError(
                    f"{prepared_path}:{line_number}: duplicate id {record_id!r}"
                )
            ids.add(record_id)
            count += 1
    if count == 0:
        raise DataPreparationError(f"{prepared_path} contains no records")
    return {"record_count": count, "sha256": digest.hexdigest()}


def _load_rows(source: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    source_kind = _require_string(source, "kind")
    if source_kind == "local":
        yield from _load_local_rows(source)
        return
    if source_kind == "huggingface":
        yield from _load_huggingface_rows(source)
        return
    raise DataPreparationError(
        f"unsupported source.kind={source_kind!r}; expected 'local' or 'huggingface'"
    )


def _load_local_rows(source: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    path = Path(os.path.expandvars(_require_string(source, "path"))).expanduser()
    if not path.is_file():
        raise DataPreparationError(f"local source does not exist: {path}")
    data_format = str(source.get("format", "auto")).lower()
    if data_format == "auto":
        data_format = path.suffix.lower().lstrip(".")
    if data_format == "jsonl":
        with path.open("r", encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataPreparationError(f"{path}:{line_number}: {exc}") from exc
                yield _ensure_row(row, f"{path}:{line_number}")
        return
    if data_format == "json":
        with path.open("r", encoding="utf-8") as source_file:
            payload = json.load(source_file)
        if isinstance(payload, Mapping):
            data_field = str(source.get("data_field", "data"))
            payload = _field(payload, data_field)
        if not isinstance(payload, list):
            raise DataPreparationError(f"{path}: JSON source must resolve to a list")
        for index, row in enumerate(payload):
            yield _ensure_row(row, f"{path}:item {index}")
        return
    if data_format == "csv":
        with path.open("r", encoding=str(source.get("encoding", "utf-8-sig")), newline="") as source_file:
            yield from csv.DictReader(source_file)
        return
    raise DataPreparationError(f"unsupported local source format: {data_format!r}")


def _load_huggingface_rows(source: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataPreparationError(
            "Hugging Face sources require the 'datasets' package"
        ) from exc

    kwargs: dict[str, Any] = {
        "path": _require_string(source, "path"),
        "split": str(source.get("split", "test")),
    }
    if source.get("name") is not None:
        kwargs["name"] = source["name"]
    if source.get("data_files") is not None:
        kwargs["data_files"] = source["data_files"]
    if source.get("cache_dir") is not None:
        kwargs["cache_dir"] = source["cache_dir"]
    if source.get("revision") is not None:
        kwargs["revision"] = source["revision"]
    if source.get("trust_remote_code") is not None:
        kwargs["trust_remote_code"] = bool(source["trust_remote_code"])
    if source.get("streaming") is not None:
        kwargs["streaming"] = bool(source["streaming"])

    try:
        dataset = load_dataset(**kwargs)
    except Exception as exc:
        raise DataPreparationError(f"failed to load Hugging Face dataset: {exc}") from exc
    shuffle_buffer_size = source.get("shuffle_buffer_size")
    if shuffle_buffer_size is not None:
        if not bool(source.get("streaming", False)):
            raise DataPreparationError(
                "source.shuffle_buffer_size is only valid with source.streaming=true"
            )
        dataset = dataset.shuffle(
            seed=int(source.get("shuffle_seed", 0)),
            buffer_size=int(shuffle_buffer_size),
        )
    for index, row in enumerate(dataset):
        yield _ensure_row(row, f"huggingface item {index}")


def _materialize_rows(
    rows: Iterable[Mapping[str, Any]],
    source: Mapping[str, Any],
    sampling: Any,
) -> list[Mapping[str, Any]]:
    if bool(source.get("streaming", False)):
        if not isinstance(sampling, Mapping):
            raise DataPreparationError("sampling must be an object")
        if bool(sampling.get("shuffle", False)):
            raise DataPreparationError(
                "streaming sources must use source.shuffle_buffer_size instead of "
                "sampling.shuffle"
            )
        limit = sampling.get("limit")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise DataPreparationError(
                "streaming sources require a positive sampling.limit"
            )
        return list(itertools.islice(rows, limit))
    return _select_rows(list(rows), sampling)


def _select_rows(
    rows: list[Mapping[str, Any]], sampling: Any
) -> list[Mapping[str, Any]]:
    if sampling is None:
        return rows
    if not isinstance(sampling, Mapping):
        raise DataPreparationError("sampling must be an object")
    selected = list(rows)
    if bool(sampling.get("shuffle", False)):
        random.Random(int(sampling.get("seed", 0))).shuffle(selected)
    limit = sampling.get("limit")
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise DataPreparationError("sampling.limit must be a positive integer")
        selected = selected[:limit]
    return selected


def _prepare_examples(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    task_name: str,
    task_type: str,
) -> list[dict[str, Any]]:
    mapping = config.get("mapping", {})
    if not isinstance(mapping, Mapping):
        raise DataPreparationError("mapping must be an object")
    transform = str(config.get("transform", "identity"))
    records = []
    for index, row in enumerate(rows):
        try:
            if transform == "lambada_last_word":
                prompt, target = _split_lambada(row, mapping)
            elif transform == "identity":
                prompt = _render_or_field(row, mapping, "prompt", required=True)
                target = (
                    _render_or_field(row, mapping, "target", required=True)
                    if task_type == "generation"
                    else None
                )
            else:
                raise DataPreparationError(f"unsupported transform: {transform!r}")

            record_id = _record_id(row, mapping, task_name, index)
            metadata = _metadata(row, mapping)
            if task_type == "generation":
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "id": record_id,
                    "task": task_name,
                    "type": task_type,
                    "prompt": str(prompt),
                    "reference": {"text": str(target)},
                    "metadata": metadata,
                }
            else:
                choices = _choices(row, mapping)
                labels = _choice_labels(mapping, len(choices))
                answer_index = _answer_index(row, mapping, labels, len(choices))
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "id": record_id,
                    "task": task_name,
                    "type": task_type,
                    "prompt": str(prompt),
                    "choices": choices,
                    "choice_labels": labels,
                    "reference": {
                        "answer_index": answer_index,
                        "answer_label": labels[answer_index],
                    },
                    "metadata": metadata,
                }
            records.append(record)
        except (KeyError, TypeError, ValueError, DataPreparationError) as exc:
            raise DataPreparationError(f"source row {index}: {exc}") from exc
    return records


def _prepare_perplexity(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    task_name: str,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = config.get("mapping", {})
    token_config = _require_mapping(config, "tokenizer")
    if not isinstance(mapping, Mapping):
        raise DataPreparationError("mapping must be an object")
    text_field = str(mapping.get("text", "text"))
    separator = str(config.get("separator", "\n\n"))
    if tokenizer is None:
        tokenizer = _load_tokenizer(token_config)
    perplexity_mode = str(config.get("perplexity_mode", "concatenate"))
    if perplexity_mode == "document_windows":
        return _prepare_document_windows(
            rows, config, task_name, tokenizer, token_config, text_field
        )
    if perplexity_mode != "concatenate":
        raise DataPreparationError(
            "perplexity_mode must be 'concatenate' or 'document_windows'"
        )

    texts = [str(_field(row, text_field)) for row in rows]
    if bool(config.get("drop_empty", True)):
        texts = [text for text in texts if text.strip()]
    if not texts:
        raise DataPreparationError("perplexity source contains no non-empty text")

    add_special_tokens = bool(token_config.get("add_special_tokens", False))
    token_ids = _encode(tokenizer, separator.join(texts), add_special_tokens)

    sequence_length = int(config.get("sequence_length", 1024))
    stride = int(config.get("stride", sequence_length - 1))
    drop_last = bool(config.get("drop_last", True))
    if sequence_length < 2:
        raise DataPreparationError("sequence_length must be at least 2")
    if not 1 <= stride <= sequence_length:
        raise DataPreparationError("stride must be in [1, sequence_length]")

    records: list[dict[str, Any]] = []
    start = 0
    while start < len(token_ids) - 1:
        window = token_ids[start : start + sequence_length]
        if len(window) < sequence_length and drop_last:
            break
        if len(window) < 2:
            break
        score_from = (
            1 if start == 0 or stride == sequence_length else sequence_length - stride
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": f"{task_name}-{len(records):06d}",
                "task": task_name,
                "type": "perplexity",
                "tokens": window,
                "score_from": score_from,
                "metadata": {
                    "token_start": start,
                    "token_end": start + len(window),
                },
            }
        )
        if start + len(window) >= len(token_ids):
            break
        start += stride
    if not records:
        raise DataPreparationError(
            "tokenized input is shorter than sequence_length; set drop_last=false "
            "or reduce sequence_length"
        )
    tokenizer_info = {
        "path": str(token_config.get("path", getattr(tokenizer, "name_or_path", ""))),
        "revision": token_config.get("revision"),
        "class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "full_vocab_size": len(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else None,
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "add_special_tokens": add_special_tokens,
        "token_count": len(token_ids),
        "sequence_length": sequence_length,
        "stride": stride,
        "drop_last": drop_last,
    }
    return records, tokenizer_info


def _prepare_document_windows(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    task_name: str,
    tokenizer: Any,
    token_config: Mapping[str, Any],
    text_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sampling = config.get("sampling", {})
    if not isinstance(sampling, Mapping):
        raise DataPreparationError("sampling must be an object")
    sample_count = sampling.get("limit")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
        raise DataPreparationError(
            "document_windows mode requires a positive sampling.limit"
        )
    sequence_length = int(config.get("sequence_length", 1024))
    if sequence_length < 2:
        raise DataPreparationError("sequence_length must be at least 2")
    max_source_rows = int(config.get("max_source_rows", sample_count * 100))
    if max_source_rows < sample_count:
        raise DataPreparationError("max_source_rows must be at least sampling.limit")
    add_special_tokens = bool(token_config.get("add_special_tokens", False))
    rng = random.Random(int(sampling.get("seed", 0)))

    records: list[dict[str, Any]] = []
    scanned_rows = 0
    for source_index, row in enumerate(rows):
        if scanned_rows >= max_source_rows:
            break
        scanned_rows += 1
        text = str(_field(row, text_field))
        if bool(config.get("drop_empty", True)) and not text.strip():
            continue
        token_ids = _encode(tokenizer, text, add_special_tokens)
        if len(token_ids) < sequence_length:
            continue
        token_start = rng.randint(0, len(token_ids) - sequence_length)
        window = token_ids[token_start : token_start + sequence_length]
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": f"{task_name}-{len(records):06d}",
                "task": task_name,
                "type": "perplexity",
                "tokens": window,
                "score_from": 1,
                "metadata": {
                    "source_row": source_index,
                    "token_start": token_start,
                    "token_end": token_start + sequence_length,
                },
            }
        )
        if len(records) == sample_count:
            break
    if len(records) != sample_count:
        raise DataPreparationError(
            f"only found {len(records)} documents with at least {sequence_length} tokens "
            f"after scanning {scanned_rows} rows; requested {sample_count}"
        )
    tokenizer_info = {
        "path": str(token_config.get("path", getattr(tokenizer, "name_or_path", ""))),
        "revision": token_config.get("revision"),
        "class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "full_vocab_size": len(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else None,
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "add_special_tokens": add_special_tokens,
        "sample_count": sample_count,
        "sequence_length": sequence_length,
        "mode": "document_windows",
        "scanned_source_rows": scanned_rows,
    }
    return records, tokenizer_info


def _encode(tokenizer: Any, text: str, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if len(token_ids) > 0 and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise DataPreparationError("tokenizer returned more than one token sequence")
        token_ids = token_ids[0]
    return [int(token) for token in token_ids]


def _load_tokenizer(token_config: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise DataPreparationError("PPL preparation requires the 'transformers' package") from exc
    path = _require_string(token_config, "path")
    kwargs = {
        "trust_remote_code": bool(token_config.get("trust_remote_code", False)),
        "use_fast": bool(token_config.get("use_fast", True)),
    }
    revision = token_config.get("revision")
    if revision is not None:
        kwargs["revision"] = revision
    cache_dir = token_config.get("cache_dir")
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(os.path.expandvars(str(cache_dir))).expanduser())
    if token_config.get("local_files_only") is not None:
        kwargs["local_files_only"] = bool(token_config["local_files_only"])
    try:
        return AutoTokenizer.from_pretrained(path, **kwargs)
    except Exception as exc:
        raise DataPreparationError(f"failed to load tokenizer {path!r}: {exc}") from exc


def _split_lambada(
    row: Mapping[str, Any], mapping: Mapping[str, Any]
) -> tuple[str, str]:
    text = str(_field(row, str(mapping.get("text", "text"))))
    match = re.match(r"^(.*\S)(\s+)(\S+)\s*$", text, flags=re.DOTALL)
    if match is None:
        raise DataPreparationError("cannot split text into a prompt and final word")
    return match.group(1) + match.group(2), match.group(3)


def _render_or_field(
    row: Mapping[str, Any],
    mapping: Mapping[str, Any],
    name: str,
    required: bool,
) -> Any:
    template_key = f"{name}_template"
    if template_key in mapping:
        return str(mapping[template_key]).format_map(_StrictFormatDict(row))
    if name in mapping:
        return _field(row, str(mapping[name]))
    if required:
        raise DataPreparationError(f"mapping.{name} or mapping.{template_key} is required")
    return None


def _choices(row: Mapping[str, Any], mapping: Mapping[str, Any]) -> list[str]:
    choice_fields = mapping.get("choice_fields")
    if choice_fields is not None:
        if not isinstance(choice_fields, list) or len(choice_fields) < 2:
            raise DataPreparationError("mapping.choice_fields must contain at least two fields")
        return [str(_field(row, str(field))) for field in choice_fields]
    choices_field = mapping.get("choices", "choices")
    choices = _field(row, str(choices_field))
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        raise DataPreparationError("mapped choices must be a list")
    return [str(choice) for choice in choices]


def _choice_labels(mapping: Mapping[str, Any], count: int) -> list[str]:
    labels = mapping.get("choice_labels")
    if labels is None:
        if count > 26:
            raise DataPreparationError("automatic choice labels support at most 26 choices")
        return [chr(ord("A") + index) for index in range(count)]
    if not isinstance(labels, list) or len(labels) != count:
        raise DataPreparationError("mapping.choice_labels length must match choices")
    return [str(label) for label in labels]


def _answer_index(
    row: Mapping[str, Any],
    mapping: Mapping[str, Any],
    labels: Sequence[str],
    choice_count: int,
) -> int:
    answer_field = str(mapping.get("answer", "answer"))
    answer = _field(row, answer_field)
    if isinstance(answer, int) and not isinstance(answer, bool):
        index = answer - int(mapping.get("answer_index_base", 0))
    else:
        answer_text = str(answer).strip()
        try:
            index = list(labels).index(answer_text)
        except ValueError:
            if answer_text.isdigit():
                index = int(answer_text) - int(mapping.get("answer_index_base", 0))
            else:
                raise DataPreparationError(
                    f"answer {answer_text!r} is neither a choice label nor an integer"
                )
    if not 0 <= index < choice_count:
        raise DataPreparationError(f"answer index {index} is outside choices")
    return index


def _metadata(row: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict[str, Any]:
    metadata_fields = mapping.get("metadata_fields", [])
    if not isinstance(metadata_fields, list):
        raise DataPreparationError("mapping.metadata_fields must be a list")
    return {str(field): _json_compatible(_field(row, str(field))) for field in metadata_fields}


def _record_id(
    row: Mapping[str, Any], mapping: Mapping[str, Any], task_name: str, index: int
) -> str:
    id_field = mapping.get("id")
    if id_field is None:
        return f"{task_name}-{index:06d}"
    return str(_field(row, str(id_field)))


def _field(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"field {path!r} not found")
        value = value[part]
    return value


def _validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise DataPreparationError("preparation produced no records")
    ids: set[str] = set()
    for index, record in enumerate(records):
        try:
            validate_record(record)
        except DataValidationError as exc:
            raise DataPreparationError(f"prepared record {index}: {exc}") from exc
        if record["id"] in ids:
            raise DataPreparationError(f"duplicate prepared id: {record['id']!r}")
        ids.add(record["id"])


def _write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output_file:
            for record in records:
                line = (
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                output_file.write(line)
                digest.update(line)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _source_manifest(source: Mapping[str, Any]) -> dict[str, Any]:
    safe_keys = (
        "kind",
        "path",
        "format",
        "name",
        "split",
        "revision",
        "data_files",
        "streaming",
        "shuffle_seed",
        "shuffle_buffer_size",
    )
    manifest = {
        key: _json_compatible(source[key]) for key in safe_keys if key in source
    }
    if source.get("kind") == "local" and isinstance(source.get("path"), str):
        source_path = Path(source["path"]).expanduser()
        if source_path.is_file():
            manifest["size_bytes"] = source_path.stat().st_size
            manifest["sha256"] = _sha256_file(source_path)
    return manifest


def _model_bindings(bindings: Any) -> list[dict[str, Any]]:
    if not isinstance(bindings, list) or not bindings:
        raise DataPreparationError("model_bindings must be a non-empty list")
    result = []
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            raise DataPreparationError(f"model_bindings[{index}] must be an object")
        name = _require_string(binding, "name")
        artifacts = {}
        for key in ("rknn", "weight", "tokenizer_gguf", "embed"):
            value = binding.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                raise DataPreparationError(
                    f"model_bindings[{index}].{key} must be a path string"
                )
            artifact_path = Path(os.path.expandvars(value)).expanduser()
            if not artifact_path.is_file():
                raise DataPreparationError(f"model artifact does not exist: {artifact_path}")
            artifact = {
                "path": value,
                "size_bytes": artifact_path.stat().st_size,
            }
            if key in {"rknn", "tokenizer_gguf"}:
                artifact["sha256"] = _sha256_file(artifact_path)
            if key == "tokenizer_gguf":
                try:
                    artifact["tokenizer_fingerprint"] = tokenizer_fingerprint(artifact_path)
                except GGUFError as exc:
                    raise DataPreparationError(
                        f"cannot inspect tokenizer GGUF {artifact_path}: {exc}"
                    ) from exc
            artifacts[key] = artifact
        if "rknn" not in artifacts or "weight" not in artifacts:
            raise DataPreparationError(
                f"model_bindings[{index}] requires rknn and weight artifacts"
            )
        result.append({"name": name, "artifacts": artifacts})
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _json_compatible(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _ensure_row(row: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise DataPreparationError(f"{location}: each source row must be an object")
    return row


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise DataPreparationError(f"{key} must be an object")
    return value


def _require_string(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataPreparationError(f"{key} must be a non-empty string")
    return value


class _StrictFormatDict(dict[str, Any]):
    def __init__(self, row: Mapping[str, Any]) -> None:
        super().__init__(row)

    def __missing__(self, key: str) -> Any:
        raise KeyError(f"template field {key!r} not found")
