from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .metrics import token_nll


@dataclass(frozen=True)
class RKNNModelFiles:
    model: Path
    weight: Path
    embedding: Path

    def validate(self) -> None:
        for path in (self.model, self.weight, self.embedding):
            if not path.is_file():
                raise FileNotFoundError(path)


@dataclass(frozen=True)
class RecordScore:
    nll_sum: float
    scored_tokens: int
    elapsed_seconds: float
    n_prefill_tokens: int
    n_decode_tokens: int

    @property
    def mean_nll(self) -> float:
        return self.nll_sum / self.scored_tokens


class RKNNPerplexityRunner:
    """Teacher-forced perplexity runner for RKNN3 LLM sessions.

    The prompt contains the unscored prefix.  For every following step the
    sampling callback reads the full vocabulary logits, accumulates the NLL of
    the gold token, and returns that same gold token to the Runtime.  This is
    exact autoregressive teacher forcing even when the exported model keeps
    only the final prefill logit.
    """

    def __init__(
        self,
        files: RKNNModelFiles,
        *,
        target: str = "rk1828",
        core_mask: int = 0xFF,
        max_context_len: int = 1024,
        device_id: str | None = None,
        logits_name: str = "logits",
        special_bos_id: int = 11,
        special_eos_id: int = 248046,
        linefeed_id: int = 198,
        verbose: bool = False,
    ) -> None:
        self.files = files
        self.target = target
        self.core_mask = core_mask
        self.max_context_len = max_context_len
        self.device_id = device_id
        self.logits_name = logits_name
        self.special_bos_id = special_bos_id
        self.special_eos_id = special_eos_id
        self.linefeed_id = linefeed_id
        self.verbose = verbose

        self._rknn: Any = None
        self._embedding: np.memmap | None = None
        self._vocab_size = 0
        self._embedding_dim = 0
        self._targets: np.ndarray | None = None
        self._target_index = 0
        self._record_nll = 0.0
        self._callback_error: BaseException | None = None
        self._callback_refs: list[Any] = []
        self.model_config: dict[str, Any] = {}
        self.sdk_version: str | None = None

    def __enter__(self) -> "RKNNPerplexityRunner":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def open(self) -> None:
        if self._rknn is not None:
            return
        self.files.validate()
        if self.max_context_len < 2:
            raise ValueError("max_context_len must be at least 2")

        try:
            from rknn3lite.api import (
                LLMGetEmbedCallback,
                LLMResultCallback,
                LLMSamplingCallback,
                RKLLMCallback,
                RKNN3Lite,
                RKNN3QueryCmd,
            )
        except ImportError as exc:
            raise RuntimeError(
                "rknn3lite is required on the target board; activate "
                "the Rknn3ToolkitLite environment"
            ) from exc

        rknn = RKNN3Lite(llm_mode=True, verbose=self.verbose)
        self._rknn = rknn
        try:
            self._check_ret(
                rknn.load_rknn(str(self.files.model), str(self.files.weight)),
                "load_rknn",
            )
            self._check_ret(
                rknn.init_runtime(
                    target=self.target,
                    core_mask=self.core_mask,
                    device_id=self.device_id,
                ),
                "init_runtime",
            )

            config = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_LLM_CONFIG)
            if config is None:
                raise RuntimeError("RKNN3_QUERY_LLM_CONFIG failed")
            self._vocab_size = int(config.vocab_size)
            self._embedding_dim = int(config.embedding_dim)
            model_max_context = int(config.max_ctx_len)
            if self.max_context_len > model_max_context:
                raise ValueError(
                    f"max_context_len={self.max_context_len} exceeds model limit "
                    f"{model_max_context}"
                )
            self.model_config = {
                "vocab_size": self._vocab_size,
                "embedding_dim": self._embedding_dim,
                "max_ctx_len": model_max_context,
                "max_position_embeddings": int(config.max_position_embeddings),
                "model_type": _decode_c_string(config.model_type),
                "task_type": int(config.task_type),
            }
            self.sdk_version = rknn.get_sdk_version()
            self._open_embedding()

            result_cb = LLMResultCallback(self._result_callback)
            sampling_cb = LLMSamplingCallback(self._sampling_callback)
            embedding_cb = LLMGetEmbedCallback(self._embedding_callback)
            userdata_object = ctypes.py_object(self)
            userdata_pointer = ctypes.cast(
                ctypes.pointer(userdata_object), ctypes.c_void_p
            )
            callback = RKLLMCallback()
            callback.result_callback = result_cb
            callback.result_userdata = userdata_pointer
            callback.sampling_callback = sampling_cb
            callback.sampling_userdata = userdata_pointer
            callback.embed_callback = embedding_cb
            callback.embed_userdata = userdata_pointer
            self._callback_refs = [
                result_cb,
                sampling_cb,
                embedding_cb,
                userdata_object,
                userdata_pointer,
                callback,
            ]

            args = [{
                "top_k": 1,
                "top_p": 1.0,
                "temperature": 1.0,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "vocab_size": self._vocab_size,
                "special_bos_id": self.special_bos_id,
                "special_eos_id": self.special_eos_id,
                "linefeed_id": self.linefeed_id,
                "skip_special_token": False,
                "ignore_eos_token": True,
                "keep_history": False,
                "max_new_tokens": self.max_context_len - 1,
                # rknn3-toolkit-lite 1.1.0 assigns this value directly to a
                # ctypes c_char_p field, so bytes are required here even
                # though the PDF API reference describes it as a string.
                "logits_name": self.logits_name.encode("utf-8"),
                "max_context_len": self.max_context_len,
            }]
            self._check_ret(rknn.init_llm_session(args, callback), "init_llm_session")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        rknn, self._rknn = self._rknn, None
        if rknn is not None:
            rknn.release()
        self._embedding = None
        self._callback_refs.clear()

    def score(self, tokens: Sequence[int], score_from: int) -> RecordScore:
        if self._rknn is None:
            raise RuntimeError("runner is not open")
        token_array = np.asarray(tokens, dtype=np.int32)
        if token_array.ndim != 1 or token_array.size < 2:
            raise ValueError("tokens must be a one-dimensional sequence of length >= 2")
        if not 1 <= score_from < token_array.size:
            raise ValueError("score_from must be in [1, len(tokens))")
        if token_array.size > self.max_context_len:
            raise ValueError(
                f"record has {token_array.size} tokens, exceeding max_context_len="
                f"{self.max_context_len}"
            )
        if int(token_array.min()) < 0 or int(token_array.max()) >= self._vocab_size:
            raise ValueError("record contains a token outside the model vocabulary")

        prefix = token_array[:score_from]
        self._targets = token_array[score_from:]
        self._target_index = 0
        self._record_nll = 0.0
        self._callback_error = None

        clear_ret = self._rknn.clear_kvcache()
        self._check_ret(clear_ret, "clear_kvcache")
        started = time.monotonic()
        result = self._rknn.session_run(
            tokens=prefix,
            keep_history=False,
            max_new_tokens=int(self._targets.size),
            prefill_only=False,
            disable_sampling=False,
        )
        elapsed = time.monotonic() - started
        if self._callback_error is not None:
            raise RuntimeError("RKNN callback failed") from self._callback_error
        if result is None or len(result) != 2:
            raise RuntimeError(f"session_run returned an invalid result: {result!r}")
        ret, stats = result
        self._check_ret(ret, "session_run")
        if self._target_index != self._targets.size:
            raise RuntimeError(
                f"sampling callback scored {self._target_index} tokens, expected "
                f"{self._targets.size}"
            )
        n_decode = int(stats[0]) if len(stats) > 0 else 0
        n_prefill = int(stats[1]) if len(stats) > 1 else 0
        return RecordScore(
            nll_sum=self._record_nll,
            scored_tokens=self._target_index,
            elapsed_seconds=elapsed,
            n_prefill_tokens=n_prefill,
            n_decode_tokens=n_decode,
        )

    def _open_embedding(self) -> None:
        expected = self._vocab_size * self._embedding_dim * np.dtype("<f2").itemsize
        actual = os.path.getsize(self.files.embedding)
        if actual != expected:
            raise ValueError(
                f"embedding size mismatch: expected {expected} bytes for "
                f"[{self._vocab_size}, {self._embedding_dim}], got {actual}"
            )
        self._embedding = np.memmap(
            self.files.embedding,
            mode="r",
            dtype="<f2",
            shape=(self._vocab_size, self._embedding_dim),
        )

    def _sampling_callback(self, userdata: int, logits: Any, logits_name: bytes) -> int:
        del userdata, logits_name
        try:
            if self._targets is None or self._target_index >= self._targets.size:
                raise RuntimeError("sampling callback received more steps than targets")
            target = int(self._targets[self._target_index])
            raw = ctypes.cast(logits, ctypes.POINTER(ctypes.c_uint16))
            values = np.ctypeslib.as_array(raw, shape=(self._vocab_size,)).view(np.float16)
            self._record_nll += token_nll(values, target)
            self._target_index += 1
            return target
        except BaseException as exc:
            self._callback_error = exc
            return -1

    def _embedding_callback(
        self,
        userdata: int,
        tokens: Any,
        num_tokens: int,
        output: int,
        output_len: int,
    ) -> int:
        del userdata
        try:
            if self._embedding is None:
                raise RuntimeError("embedding table is not open")
            token_ids = np.ctypeslib.as_array(tokens, shape=(num_tokens,))
            expected_elements = int(num_tokens) * self._embedding_dim
            if int(output_len) != expected_elements * np.dtype(np.float16).itemsize:
                raise ValueError(
                    f"embedding callback buffer is {output_len} bytes, expected "
                    f"{expected_elements * 2}"
                )
            if token_ids.size and (
                int(token_ids.min()) < 0 or int(token_ids.max()) >= self._vocab_size
            ):
                raise ValueError("embedding callback received an invalid token id")
            raw = ctypes.cast(output, ctypes.POINTER(ctypes.c_uint16))
            destination = np.ctypeslib.as_array(raw, shape=(expected_elements,))
            source = np.asarray(self._embedding[token_ids], dtype="<f2").reshape(-1).view(np.uint16)
            np.copyto(destination, source)
            return 0
        except BaseException as exc:
            self._callback_error = exc
            return -1

    @staticmethod
    def _result_callback(userdata: int, result: Any, state: int) -> int:
        del userdata, result, state
        return 0

    @staticmethod
    def _check_ret(ret: Any, operation: str) -> None:
        if ret != 0:
            raise RuntimeError(f"{operation} failed with ret={ret}")


def _decode_c_string(value: bytes | None) -> str | None:
    return value.decode("utf-8", errors="replace") if value else None
