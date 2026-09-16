from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from prepare_rknn_data import _load_config
from rknn_eval.data.gguf import tokenizer_fingerprint
from rknn_eval.data.prepare import (
    DataPreparationError,
    prepare_dataset,
    validate_prepared_file,
)


class FakeTokenizer:
    name_or_path = "fake-tokenizer"
    vocab_size = 256

    def __call__(self, text: str, add_special_tokens: bool = False):
        tokens = [ord(character) % self.vocab_size for character in text]
        if add_special_tokens:
            tokens.insert(0, 1)
        return {"input_ids": tokens}


class PrepareDatasetTest(unittest.TestCase):
    def test_gguf_tokenizer_fingerprint(self):
        def gguf_string(value: str) -> bytes:
            encoded = value.encode("utf-8")
            return struct.pack("<Q", len(encoded)) + encoded

        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_path = Path(temp_dir) / "tokenizer.gguf"
            tokens = ["a", "量", "<eos>"]
            payload = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
            payload += gguf_string("tokenizer.ggml.tokens")
            payload += struct.pack("<IIQ", 9, 8, len(tokens))
            payload += b"".join(gguf_string(token) for token in tokens)
            payload += gguf_string("tokenizer.ggml.eos_token_id")
            payload += struct.pack("<II", 4, 2)
            gguf_path.write_bytes(payload)

            fingerprint = tokenizer_fingerprint(gguf_path)

            self.assertEqual(fingerprint["token_count"], 3)
            self.assertEqual(fingerprint["eos_token_id"], 2)
            self.assertEqual(len(fingerprint["tokens_sha256"]), 64)

    def test_json_config_does_not_require_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            expected = {"task": {"name": "demo", "type": "generation"}}
            config_path.write_text(json.dumps(expected), encoding="utf-8")

            self.assertEqual(_load_config(str(config_path)), expected)

    def test_generation_and_manifest_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"id": "a", "prompt": "问题一", "answer": "答案一", "category": "x"},
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {"id": "b", "prompt": "问题二", "answer": "答案二", "category": "y"},
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "prepared.jsonl"
            config = {
                "task": {"name": "demo", "type": "generation"},
                "source": {"kind": "local", "path": str(source), "format": "jsonl"},
                "mapping": {
                    "id": "id",
                    "prompt": "prompt",
                    "target": "answer",
                    "metadata_fields": ["category"],
                },
                "sampling": {"shuffle": True, "seed": 7},
                "output": {"path": str(output)},
            }

            first = prepare_dataset(config)
            first_bytes = output.read_bytes()
            second = prepare_dataset(config)

            self.assertEqual(first["output_sha256"], second["output_sha256"])
            self.assertEqual(first_bytes, output.read_bytes())
            self.assertIn("sha256", first["source"])
            self.assertIn("configuration_sha256", first)
            self.assertEqual(validate_prepared_file(output)["record_count"], 2)

    def test_multiple_choice_label_is_normalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "question": "2+2=?",
                            "A": "3",
                            "B": "4",
                            "C": "5",
                            "D": "6",
                            "answer": "B",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "prepared.jsonl"
            config = {
                "task": {"name": "choice", "type": "multiple_choice"},
                "source": {"kind": "local", "path": str(source)},
                "mapping": {
                    "prompt_template": "{question}\n答案：",
                    "choice_fields": ["A", "B", "C", "D"],
                    "answer": "answer",
                },
                "output": {"path": str(output)},
            }

            prepare_dataset(config)
            record = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(record["choices"], ["3", "4", "5", "6"])
            self.assertEqual(record["reference"]["answer_index"], 1)
            self.assertEqual(record["reference"]["answer_label"], "B")

    def test_perplexity_windows_only_score_new_overlap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            source.write_text(json.dumps({"text": "abcdefghi"}) + "\n", encoding="utf-8")
            output = root / "prepared.jsonl"
            config = {
                "task": {"name": "ppl", "type": "perplexity"},
                "source": {"kind": "local", "path": str(source)},
                "mapping": {"text": "text"},
                "tokenizer": {"path": "unused", "add_special_tokens": False},
                "sequence_length": 6,
                "stride": 4,
                "drop_last": False,
                "output": {"path": str(output)},
            }

            manifest = prepare_dataset(config, tokenizer=FakeTokenizer())
            records = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(manifest["tokenizer"]["token_count"], 9)
            self.assertEqual([record["score_from"] for record in records], [1, 2])
            self.assertEqual([len(record["tokens"]) for record in records], [6, 5])
            scored_global_indices = []
            for record in records:
                start = record["metadata"]["token_start"]
                scored_global_indices.extend(
                    start + index
                    for index in range(record["score_from"], len(record["tokens"]))
                )
            self.assertEqual(scored_global_indices, list(range(1, 9)))

    def test_perplexity_non_overlapping_blocks_match_eval_ppl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            source.write_text(json.dumps({"text": "abcdefghijklmn"}) + "\n", encoding="utf-8")
            output = root / "prepared.jsonl"
            config = {
                "task": {"name": "ppl", "type": "perplexity"},
                "source": {"kind": "local", "path": str(source)},
                "mapping": {"text": "text"},
                "tokenizer": {"path": "unused", "add_special_tokens": False},
                "sequence_length": 6,
                "stride": 6,
                "drop_last": True,
                "output": {"path": str(output)},
            }

            manifest = prepare_dataset(config, tokenizer=FakeTokenizer())
            records = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(manifest["tokenizer"]["token_count"], 14)
            self.assertEqual([record["metadata"]["token_start"] for record in records], [0, 6])
            self.assertEqual([record["score_from"] for record in records], [1, 1])
            self.assertEqual(sum(len(record["tokens"]) - 1 for record in records), 10)

    def test_document_windows_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps({"text": text})
                    for text in ["short", "abcdefgh", "ijklmnop", "qrstuvwx"]
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "prepared.jsonl"
            config = {
                "task": {"name": "c4", "type": "perplexity"},
                "source": {"kind": "local", "path": str(source)},
                "mapping": {"text": "text"},
                "tokenizer": {"path": "unused", "add_special_tokens": False},
                "perplexity_mode": "document_windows",
                "sequence_length": 6,
                "sampling": {"limit": 2, "seed": 11},
                "output": {"path": str(output)},
            }

            first = prepare_dataset(config, tokenizer=FakeTokenizer())
            first_bytes = output.read_bytes()
            second = prepare_dataset(config, tokenizer=FakeTokenizer())
            records = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(first["output_sha256"], second["output_sha256"])
            self.assertEqual(first_bytes, output.read_bytes())
            self.assertEqual(len(records), 2)
            self.assertTrue(all(len(record["tokens"]) == 6 for record in records))
            self.assertTrue(all(record["score_from"] == 1 for record in records))

    def test_lambada_transform_preserves_separator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            source.write_text(
                json.dumps({"text": "The final word is answer"}) + "\n",
                encoding="utf-8",
            )
            output = root / "prepared.jsonl"
            config = {
                "task": {"name": "lambada", "type": "generation"},
                "source": {"kind": "local", "path": str(source)},
                "transform": "lambada_last_word",
                "mapping": {"text": "text"},
                "output": {"path": str(output)},
            }

            prepare_dataset(config)
            record = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(record["prompt"], "The final word is ")
            self.assertEqual(record["reference"]["text"], "answer")

    def test_duplicate_source_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    [
                        {"id": "same", "prompt": "p1", "target": "t1"},
                        {"id": "same", "prompt": "p2", "target": "t2"},
                    ]
                ),
                encoding="utf-8",
            )
            config = {
                "task": {"name": "demo", "type": "generation"},
                "source": {"kind": "local", "path": str(source)},
                "mapping": {"id": "id", "prompt": "prompt", "target": "target"},
                "output": {"path": str(root / "prepared.jsonl")},
            }

            with self.assertRaisesRegex(DataPreparationError, "duplicate prepared id"):
                prepare_dataset(config)


if __name__ == "__main__":
    unittest.main()
