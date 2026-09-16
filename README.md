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

当前分支已完成 RKNN3 板端评估的数据准备层和通用文本 LLM C++
评估器，首批目标模型为：

- `Qwen3.5-4B-Q2N-W4A16-G32`；
- `Qwen3.5-4B-rknn-official`。

支持 WikiText-2/C4 困惑度、TTFT、Prefill/Decode 吞吐、RK1828 设备内存
分项、RK3588 进程 RSS 和最大上下文查询。完整设计见
[`docs/RKNN_BOARD_EVALUATION_DESIGN.md`](docs/RKNN_BOARD_EVALUATION_DESIGN.md)。

### 环境

主机端使用：

```bash
conda activate rknn_quant_bench_env
pip install -r requirements_rknn.txt
```

板端 Python 原型使用 `/root/miniforge3/envs/Rknn3ToolkitLite/bin/python`；正式
全量评估使用 C++ runner，不依赖板端 Python 环境。

数据工具支持本地 JSON/JSONL/CSV 和 Hugging Face 数据源，输出经过校验的
JSONL 及 SHA-256 manifest。读取 YAML 配置需要 PyYAML，已经包含在
`requirements_rknn.txt` 中。

### 板端模型目录

当前 RK3588+RK1828 板上的模型目录为：

```text
/userdata/llm_demo/rknn_Qwen3_5_demo/
├── model_4b_q2n_w4a16_g32/
│   ├── Qwen3.5-4B-Q2N-W4A16-G32.rknn
│   ├── Qwen3.5-4B-Q2N-W4A16-G32.weight
│   ├── Qwen3.5-4B-Q2N-W4A16-G32.tokenizer.gguf
│   └── Qwen3.5-4B-Q2N-W4A16-G32.embed.bin
└── model_4b/
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
  --config configs/rknn/wikitext2_qwen35_4b_ppl_seq2048.yaml
python prepare_rknn_data.py prepare \
  --config configs/rknn/c4_qwen35_4b_ppl.yaml
```

验证已有数据：

```bash
python prepare_rknn_data.py validate \
  --input data/rknn/qwen35_4b/wikitext2_ppl_seq2048.jsonl
python prepare_rknn_data.py validate \
  --input data/rknn/qwen35_4b/c4_ppl_256x1024.jsonl
```

数据规格：

| 数据集 | 处理方式 | 记录数 | 计分 token 数 |
| --- | --- | ---: | ---: |
| WikiText-2 test | 对齐 `eval_ppl.py`，145 个不重叠 2048-token 块 | 145 | 296,815 |
| C4 validation | 流式随机文档窗口，seed 0 | 256 | 261,888 |

C4 使用 streaming 模式，不需要完整下载数据集。两份模型的 GGUF token 数组
指纹一致，因此在关闭 special tokens 的 PPL 评估中可共用同一份 token 数据。

### 构建通用 C++ 评估器

评估器源码位于 `rknn_eval/cpp/`，复用
`~/rknn_proj/rknn3-model-zoo-1.1.0` 的 RKNN3 API、Tokenizer 和运行库，默认使用：

```text
/home/ilearn-xyf/sdk/rk1828/
└── gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/
```

构建命令：

```bash
export RKNN3_MODEL_ZOO_ROOT="$HOME/rknn_proj/rknn3-model-zoo-1.1.0"
export RK1828_TOOLCHAIN_ROOT="/home/ilearn-xyf/sdk/rk1828/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu"
./rknn_eval/cpp/build-linux.sh
```

脚本会使用 `${RK1828_TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-{gcc,g++}`。也可按
model-zoo 的约定，用 `GCC_COMPILER` 直接覆盖完整工具链前缀。

产物为：

```text
build/rknn_llm_ppl_eval_rk3588_aarch64/install/rknn_llm_ppl_eval
```

复制程序和数据到板端：

```bash
adb shell mkdir -p /userdata/ptq-bench-rknn/data /userdata/ptq-bench-rknn/results
adb push build/rknn_llm_ppl_eval_rk3588_aarch64/install/rknn_llm_ppl_eval \
  /userdata/ptq-bench-rknn/rknn_llm_ppl_eval
adb push data/rknn/qwen35_4b/wikitext2_ppl_seq2048.jsonl \
  /userdata/ptq-bench-rknn/data/
adb push data/rknn/qwen35_4b/c4_ppl_256x1024.jsonl \
  /userdata/ptq-bench-rknn/data/
```

板端已在 `/usr/lib` 安装 RKNN3 1.1.0 运行库；若目标板未安装，可同时复制构建
产物 `install/lib/`。

### 运行评估

官方模型的单条冒烟测试：

```bash
adb shell /userdata/ptq-bench-rknn/rknn_llm_ppl_eval \
  --model /userdata/llm_demo/rknn_Qwen3_5_demo/model_4b/Qwen3.5-4B.rknn \
  --weight /userdata/llm_demo/rknn_Qwen3_5_demo/model_4b/Qwen3.5-4B.weight \
  --tokenizer /userdata/llm_demo/rknn_Qwen3_5_demo/model_4b/Qwen3.5-4B.tokenizer.gguf \
  --embedding /userdata/llm_demo/rknn_Qwen3_5_demo/model_4b/Qwen3.5-4B.embed.bin \
  --data /userdata/ptq-bench-rknn/data/wikitext2_ppl_seq2048.jsonl \
  --output /userdata/ptq-bench-rknn/results/official_wikitext2.jsonl \
  --model-name Qwen3.5-4B-rknn-official \
  --logits-name logits --core-mask 0xff --max-context-len 4096 \
  --scoring-threads 4 --perf-prefill-tokens 512 --perf-decode-tokens 128 \
  --limit 1 --no-resume
```

Q2N-W4A16-G32 只需替换四个模型文件路径和 `--model-name`。去掉 `--limit 1`
与 `--no-resume` 即可全量运行；默认会读取已有 JSONL 并按记录 ID 断点续跑。
C4 使用相同命令，仅替换 `--data` 和 `--output`。

正式吞吐测试建议与 PPL 解耦，预热 3 次并记录 10 个原始样本及 p50/p90：

```bash
# 模型文件参数与上例相同
adb shell /userdata/ptq-bench-rknn/rknn_llm_ppl_eval \
  --model MODEL.rknn --weight MODEL.weight \
  --tokenizer MODEL.tokenizer.gguf --embedding MODEL.embed.bin \
  --data /userdata/ptq-bench-rknn/data/wikitext2_ppl_seq2048.jsonl \
  --output /userdata/ptq-bench-rknn/results/MODEL_perf.jsonl \
  --model-name MODEL --core-mask 0xff --max-context-len 4096 \
  --perf-prefill-tokens 512 --perf-decode-tokens 128 \
  --perf-warmup 3 --perf-repeat 10 --limit 0 --no-resume
```

`--limit 0` 表示只运行性能探针，不写入 PPL 记录；summary 中 PPL 字段为
`null`。比较不同模型时应保持输入 token、核心掩码、Prefill/Decode 长度、预热和
重复次数完全一致，并在同一温控/频率条件下运行。

仓库提供完整串行队列 `rknn_eval/cpp/run-board-full.sh`，依次执行两个模型的
128/512/1024 Prefill 性能测试、WikiText-2 全量 PPL 和 C4 固定 256 窗口 PPL：

```bash
adb push rknn_eval/cpp/run-board-full.sh /userdata/ptq-bench-rknn/
adb shell chmod +x /userdata/ptq-bench-rknn/run-board-full.sh
adb shell 'cd /userdata/ptq-bench-rknn; \
  nohup setsid ./run-board-full.sh > logs/full_eval.log 2>&1 </dev/null &'
```

运行状态写入 `results/full_eval.status`，日志写入 `logs/full_eval.log`。PPL 输出支持
断点续跑；已有非空性能 summary 会被跳过，如需重测应先移走对应性能结果。

只评估 WikiText-2 和 1024-token Prefill 时：

```bash
adb shell 'cd /userdata/ptq-bench-rknn; \
  export PERF_PREFILL_LENGTHS=1024 RUN_C4=0; \
  nohup setsid ./run-board-full.sh > logs/full_eval.log 2>&1 </dev/null &'
```

`PERF_PREFILL_LENGTHS` 控制性能档位，`RUN_WIKITEXT` 和 `RUN_C4` 控制数据集；值为
`1` 时启用。已有 PPL JSONL 会按记录 ID 继续运行。

每条结果符合
[`schemas/rknn_eval_result.schema.json`](schemas/rknn_eval_result.schema.json)，最终还会生成
`<output>.summary.json`。指标口径如下：

- `perplexity`：sampling callback 对真实下一个 token 做 teacher forcing 得到的精确 PPL；
- `prefill_tokens_per_second`：独立 512-token Prefill 基准；
- `model_decode_tokens_per_second_estimate`：扣除 CPU 全词表 NLL 归约耗时后的模型 Decode 吞吐；
- `decode_tokens_per_second` / `eval_tokens_per_second`：包含 PPL 评分开销的端到端吞吐；
- `memory.allocation`：RK1828 各核 command/weight/internal/KV cache 及合计；
- `memory.host*`：RK3588 进程当前和峰值 RSS；
- `model_config.max_ctx_len`、`attention_kvcache_lengths` 和
  `session_n_max_tokens_after_run`：导出模型、KV cache 与实跑 Session 的上下文上限。

当前单样本链路验收结果如下：

| 模型 | PPL（1 条） | Prefill 512 | 模型 Decode 估计 | RK1828 分配 | 上下文上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-rknn-official | 6.94070 | 743.94 tok/s | 41.48 tok/s | 3.027 GB | 4096 |
| Qwen3.5-4B-Q2N-W4A16-G32 | 6.96426 | 640.79 tok/s | 40.74 tok/s | 3.027 GB | 4096 |

这些数值只验证评分、性能、内存和上下文采集链路，不代表最终模型优劣。正式结论
必须跑完整 WikiText-2/C4，并对性能执行预热和多次重复。

### 扩展到其他模型

`generic_text` adapter 不依赖 Qwen3.5，适用于支持 raw token 输入、标准 logits
sampling callback 的 decoder-only RKNN3 模型：

- logits 节点不同可用 `--logits-name`（例如 LFM demo 使用 `output`）；
- tied embedding 模型用 `--embedding -`；
- tokenizer、词表、embedding 维度和最大上下文均在运行时校验；
- 需要额外 `input_callback` 或多模态输入的模型（例如 Gemma4）应增加专用 adapter，
  不应直接套用 `generic_text` 得出比较结果。

### 测试

```bash
python -m unittest tests/test_rknn_data_prepare.py tests/test_rknn_board_metrics.py -v
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
