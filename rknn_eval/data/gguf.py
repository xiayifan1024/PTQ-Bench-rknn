from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any, BinaryIO


class GGUFError(ValueError):
    """Raised when a GGUF metadata file cannot be parsed."""


_SCALAR_FORMATS = {
    0: "B",   # uint8
    1: "b",   # int8
    2: "H",   # uint16
    3: "h",   # int16
    4: "I",   # uint32
    5: "i",   # int32
    6: "f",   # float32
    7: "?",   # bool
    10: "Q",  # uint64
    11: "q",  # int64
    12: "d",  # float64
}


def read_gguf_metadata(path: str | Path) -> dict[str, Any]:
    """Read GGUF key/value metadata without loading tensor payloads."""

    with Path(path).open("rb") as gguf_file:
        if _read_exact(gguf_file, 4) != b"GGUF":
            raise GGUFError("not a GGUF file")
        version = _read_scalar(gguf_file, "I")
        if version not in {2, 3}:
            raise GGUFError(f"unsupported GGUF version: {version}")
        _tensor_count = _read_scalar(gguf_file, "Q")
        metadata_count = _read_scalar(gguf_file, "Q")
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _read_string(gguf_file)
            value_type = _read_scalar(gguf_file, "I")
            metadata[key] = _read_value(gguf_file, value_type)
    return metadata


def tokenizer_fingerprint(path: str | Path) -> dict[str, Any]:
    metadata = read_gguf_metadata(path)
    tokens = metadata.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise GGUFError("tokenizer.ggml.tokens is missing or invalid")
    digest = hashlib.sha256()
    for token in tokens:
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return {
        "token_count": len(tokens),
        "tokens_sha256": digest.hexdigest(),
        "model": metadata.get("tokenizer.ggml.model"),
        "pre": metadata.get("tokenizer.ggml.pre"),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "padding_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
    }


def _read_value(stream: BinaryIO, value_type: int) -> Any:
    if value_type in _SCALAR_FORMATS:
        return _read_scalar(stream, _SCALAR_FORMATS[value_type])
    if value_type == 8:
        return _read_string(stream)
    if value_type == 9:
        element_type = _read_scalar(stream, "I")
        length = _read_scalar(stream, "Q")
        return [_read_value(stream, element_type) for _ in range(length)]
    raise GGUFError(f"unsupported GGUF metadata value type: {value_type}")


def _read_string(stream: BinaryIO) -> str:
    length = _read_scalar(stream, "Q")
    try:
        return _read_exact(stream, length).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFError("invalid UTF-8 GGUF string") from exc


def _read_scalar(stream: BinaryIO, scalar_format: str) -> Any:
    size = struct.calcsize("<" + scalar_format)
    return struct.unpack("<" + scalar_format, _read_exact(stream, size))[0]


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    value = stream.read(length)
    if len(value) != length:
        raise GGUFError("unexpected end of GGUF file")
    return value
