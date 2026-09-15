# 🔬 PTQ-Bench

This repository contains the evaluation codes for the paper **[Benchmarking Post-Training Quantization in LLMs: Comprehensive Taxonomy, Unified Evaluation, and Comparative Analysis]**. Each method (**GPTQ**, **AWQ**, **OmniQuant**, and **QuIP**) is modularized, configurable via YAML, and supports streamlined evaluation via a common launcher.

---

## 🚀 Usage

### 1. Environment Setup

```bash
conda create -n quant-bench python=3.10
conda activate quant-bench
pip install -r requirements.txt
```

> You should install the Mamba and AWQ environments separately by following their official repositories.

---

### 2. Run Quantization

Use the launcher `run_quant.py` with `--method` and `--config`:

```bash
python run_quant.py --method gptq --config configs/gptq.yaml
python run_quant.py --method omniquant --config configs/omniquant.yaml
python run_quant.py --method quip --config configs/quip.yaml
python run_quant.py --method awq --config configs/awq.yaml
```

---

### 3. Example Config: `configs/gptq.yaml`

```yaml
model_path: /PATH/TO/llama-7b
dataset: c4
wbits: 2
save_path: /PATH/TO/GPTQ/llama-7b-w2
act_order: true
CUDA_VISIBLE_DEVICES: "1"
```

---

## 4. Perplexity Evaluation

1. Save the quantized model weights.
2. Run the following command in your terminal:

```bash
python eval_ppl.py --model /PATH/TO/GPTQ/llama-7b-w2
```

---

## 5. Evaluation of Zero-shot Tasks

We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate zero-shot performance. To run an evaluation, you can use a script like the following:

```bash
TASKS="truthfulqa,hellaswag,winogrande,race,piqa,mmlu,hellaswag,arc_easy,arc_challenge,lambada,gsm8k,ceval-valid"
CUDA_VISIBLE_DEVICES=5 lm_eval --model hf \
        --model_args pretrained=/PATH/TO/GPTQ/llama-7b-w2 \
        --tasks $TASKS \
        --device cuda:0 \
        --batch_size auto:4 \
        --output ./results/GPTQ/llama-7b-w2
```

## 6. Evaluation of Multi-Modal Tasks

We use the official repository to evaluate LLaVA and VILA. For details, refer to the [Evaluation Guide](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

## RKNN3 板端评估（当前版本）

当前分支已完成 RKNN3 板端评估的数据准备层，目标模型为：

- `Qwen3.5-4B-Q2N-W4A16-G32`；
- `Qwen3.5-4B-rknn-official`。

本版本暂不包含板端 Session 推理执行器；已完成 WikiText-2/C4 数据生成、
模型与 tokenizer 绑定、数据校验和可复现 manifest。完整设计见
[`docs/RKNN_BOARD_EVALUATION_DESIGN.md`](docs/RKNN_BOARD_EVALUATION_DESIGN.md)。

### 环境

```bash
conda activate rknn-eval
pip install -r requirements_rknn.txt
```

数据工具支持本地 JSON/JSONL/CSV 和 Hugging Face 数据源，输出经过校验的
JSONL 及 SHA-256 manifest。读取 YAML 配置需要 PyYAML，已经包含在
`requirements_rknn.txt` 中。

### 模型目录

配置使用 `RKNN_MODEL_ROOT` 定位模型，默认使用方式如下：

```bash
export RKNN_MODEL_ROOT="$HOME/models"
```

目录应包含：

```text
$RKNN_MODEL_ROOT/
├── Qwen3.5-4B-Q2N-W4A16-G32/
│   ├── Qwen3.5-4B-Q2N-W4A16-G32.rknn
│   ├── Qwen3.5-4B-Q2N-W4A16-G32.weight
│   ├── Qwen3.5-4B-Q2N-W4A16-G32.tokenizer.gguf
│   └── Qwen3.5-4B-Q2N-W4A16-G32.embed.bin
└── Qwen3.5-4B-rknn-official/
    ├── Qwen3.5-4B.rknn
    ├── Qwen3.5-4B.weight
    ├── Qwen3.5-4B.tokenizer.gguf
    └── Qwen3.5-4B.embed.bin
```

### 数据准备

```bash
python prepare_rknn_data.py prepare --config configs/rknn/data.example.yaml
python prepare_rknn_data.py validate --input data/rknn/demo_generation.jsonl
```

支持三类统一记录：

- `generation`：prompt 与参考答案；
- `multiple_choice`：prompt、候选项及归一化答案索引；
- `perplexity`：token id 窗口及 `score_from`，避免重叠窗口重复计分。

完整交换格式见
[`schemas/rknn_eval_record.schema.json`](schemas/rknn_eval_record.schema.json)。

PPL 使用的 Hugging Face tokenizer 必须与 RKNN `tokenizer.gguf` 同源。
本项目会解析 GGUF 元数据并把 token 数组指纹写入 manifest。

重新生成 Qwen3.5-4B 数据：

```bash
python prepare_rknn_data.py prepare \
  --config configs/rknn/wikitext2_qwen35_4b_ppl.yaml
python prepare_rknn_data.py prepare \
  --config configs/rknn/c4_qwen35_4b_ppl.yaml
```

验证已有数据：

```bash
python prepare_rknn_data.py validate \
  --input data/rknn/qwen35_4b/wikitext2_ppl_seq1024.jsonl
python prepare_rknn_data.py validate \
  --input data/rknn/qwen35_4b/c4_ppl_256x1024.jsonl
```

数据规格：

| 数据集 | 处理方式 | 记录数 | 计分 token 数 |
| --- | --- | ---: | ---: |
| WikiText-2 test | 全文拼接，窗口 1024，步长 1023 | 290 | 296,670 |
| C4 validation | 流式随机文档窗口，seed 0 | 256 | 261,888 |

C4 使用 streaming 模式，不需要完整下载数据集。两份模型的 GGUF token 数组
指纹一致，因此在关闭 special tokens 的 PPL 评估中可共用同一份 token 数据。

### 测试

```bash
python -m unittest tests/test_rknn_data_prepare.py -v
```

## Contributors

- OpenAI Codex — RKNN3 板端评估方案、数据准备管线、数据校验与文档。

## Related Projects

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://github.com/IST-DASLab/gptq)
- [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://github.com/OpenGVLab/OmniQuant)
- [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://github.com/Cornell-RelaxML/QuIP)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)
- [Visual Instruction Tuning](https://github.com/haotian-liu/LLaVA)
- [VILA: On Pre-training for Visual Language Models](https://github.com/NVlabs/VILA)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://github.com/state-spaces/mamba)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
