# Qwen3.5-4B RKNN evaluation data

These files are shared by the Q2N-W4A16-G32 and official RKNN models. Both
board tokenizer GGUF files contain the same ordered 248,320-token array
(`tokens_sha256=b7f4906b5bf6a845baf3f41fdcdcd70f0f2f234eb702aabb830ce1536604e5d6`).

| Dataset | Preparation | Records | Scored tokens | JSONL SHA-256 |
| --- | --- | ---: | ---: | --- |
| WikiText-2 test | Concatenate non-empty text, length 1024, stride 1023 | 290 | 296,670 | `7231aa89ae64fc14aed6e6ec6c2d296c3087078f951f63ceaa5194efff9626fa` |
| C4 validation | Streaming shuffle seed 0, 256 document windows of length 1024 | 256 | 261,888 | `f99920160939b99da3b634ed6a01bb5a622ea25944c98c8b393c86b8b3679a71` |

PPL consumers must only score token positions starting at each record's
`score_from` value. This prevents overlap from being counted twice.
