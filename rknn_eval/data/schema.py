from __future__ import annotations

from typing import Any, Mapping


SCHEMA_VERSION = 1
SUPPORTED_TYPES = {"generation", "multiple_choice", "perplexity"}


class DataValidationError(ValueError):
    """Raised when a prepared evaluation record is invalid."""


def validate_record(record: Mapping[str, Any]) -> None:
    required = {"schema_version", "id", "task", "type"}
    missing = sorted(required - record.keys())
    if missing:
        raise DataValidationError(f"missing required fields: {', '.join(missing)}")

    if record["schema_version"] != SCHEMA_VERSION:
        raise DataValidationError(
            f"unsupported schema_version={record['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if not isinstance(record["id"], str) or not record["id"].strip():
        raise DataValidationError("id must be a non-empty string")
    if not isinstance(record["task"], str) or not record["task"].strip():
        raise DataValidationError("task must be a non-empty string")

    record_type = record["type"]
    if record_type not in SUPPORTED_TYPES:
        raise DataValidationError(
            f"unsupported type={record_type!r}; expected one of {sorted(SUPPORTED_TYPES)}"
        )

    if record_type == "generation":
        _validate_generation(record)
    elif record_type == "multiple_choice":
        _validate_multiple_choice(record)
    else:
        _validate_perplexity(record)

    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise DataValidationError("metadata must be an object")


def _validate_generation(record: Mapping[str, Any]) -> None:
    if not isinstance(record.get("prompt"), str):
        raise DataValidationError("generation prompt must be a string")
    reference = record.get("reference")
    if not isinstance(reference, Mapping) or not isinstance(reference.get("text"), str):
        raise DataValidationError("generation reference.text must be a string")


def _validate_multiple_choice(record: Mapping[str, Any]) -> None:
    if not isinstance(record.get("prompt"), str):
        raise DataValidationError("multiple_choice prompt must be a string")
    choices = record.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) < 2
        or not all(isinstance(choice, str) for choice in choices)
    ):
        raise DataValidationError("multiple_choice choices must contain at least two strings")
    reference = record.get("reference")
    if not isinstance(reference, Mapping):
        raise DataValidationError("multiple_choice reference must be an object")
    answer_index = reference.get("answer_index")
    if not isinstance(answer_index, int) or isinstance(answer_index, bool):
        raise DataValidationError("multiple_choice reference.answer_index must be an integer")
    if not 0 <= answer_index < len(choices):
        raise DataValidationError("multiple_choice answer_index is outside choices")


def _validate_perplexity(record: Mapping[str, Any]) -> None:
    tokens = record.get("tokens")
    if (
        not isinstance(tokens, list)
        or len(tokens) < 2
        or not all(
            isinstance(token, int) and not isinstance(token, bool) and token >= 0
            for token in tokens
        )
    ):
        raise DataValidationError(
            "perplexity tokens must contain at least two non-negative integer ids"
        )
    score_from = record.get("score_from")
    if not isinstance(score_from, int) or isinstance(score_from, bool):
        raise DataValidationError("perplexity score_from must be an integer")
    if not 1 <= score_from < len(tokens):
        raise DataValidationError("perplexity score_from must be in [1, len(tokens))")
