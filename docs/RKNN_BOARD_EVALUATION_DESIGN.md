# PTQ-Bench RKNN 板端模型推理评估适配方案

## 1. 目标与边界

本方案只解决“已有 RKNN3 模型在目标板上的推理评估”，不负责：

- CUDA/Hugging Face 与 RKNN 输出逐项对齐；
- PTQ 算法执行；
- Hugging Face -> ONNX -> RKNN 模型转换；
- RKNN 模型精度分析快照（`inference(accuracy_analysis=True)` 属于 Toolkit 连板调试能力）。

输入是已准备好的板端部署产物：

- 必需：`*.rknn`、`*.weight`；
- LLM 通常还需要：`*.tokenizer.gguf`、`*.embed.bin`；
- Tie Word Embedding 模型不需要 `embed.bin`；
- 部分模型还需要外置 RoPE 或 per-layer input 文件。

输出是一份结构化评估结果，覆盖：

1. 功能正确性：加载、初始化、单轮/多轮生成、确定性检查；
2. 任务精度：生成类任务、选择题、可选 PPL；
3. 性能：TTFT、Prefill TPS、Decode TPS/TPOT、端到端延迟；
4. 内存：Weight、Internal、KVCache、Command、进程 RSS；
5. 可复现信息：模型哈希、Runtime 版本、硬件、核心配置和采样参数。

## 2. 设计结论

板端评估采用“Host 调度 + Board Agent 执行”的两层结构：

```text
Host
  数据集/任务适配 -> 任务分片 -> SSH/HTTP/JSONL -> 汇总指标 -> result.json/report.md
                                      |
Board                                 v
  Board Agent -> RKNN3 Toolkit Lite Session -> RKNN3 Runtime -> NPU
                    |                         |
                    |                         +-> profile_ops/profile_mem
                    +-> tokenizer/embed/callback/session timing
```

选择这个结构的原因：

- 数据集、`lm-evaluation-harness` 和报告留在 Host，避免占用板端磁盘与 Python 环境；
- 模型、Tokenizer、Embedding、KVCache 和 Runtime Session 常驻板端，避免每条样本重复加载；
- 功能和精度评测优先使用 Toolkit Lite，开发成本低且具备 LLM Session/Callback/KVCache；
- 最终性能和内存以 Runtime C API 采集为准，避免把 Python/HTTP 开销误算成 NPU 推理耗时。

不建议把 `rkllm3-server` 作为唯一评估后端。它适合生成质量和 OpenAI API 兼容测试，但协议通常不能提供完整 logits、算子 profile 和准确的 NPU 内存明细。可以把它作为可选的 `server` 后端，主后端仍为 `lite`/`runtime`。

## 3. 与当前工程的衔接

当前工程的 `run_quant.py` 和各方法 `register.py` 继续只负责量化，不修改其行为。板端评估新增独立入口，避免量化环境和 RKNN 板端环境相互污染。

建议目录：

```text
PTQ-Bench-rknn/
├── run_rknn_eval.py                 # Host 统一入口
├── configs/rknn/
│   └── qwen2_5_0_5b_rk1820.yaml
├── rknn_eval/
│   ├── config.py                    # 配置解析、语义校验
│   ├── artifacts.py                 # 文件清单、哈希和部署
│   ├── transport/
│   │   ├── local.py                 # 直接在板端运行
│   │   └── ssh.py                   # Host 远程调度
│   ├── client.py                    # JSONL/HTTP 客户端
│   ├── evaluators/
│   │   ├── smoke.py
│   │   ├── generation.py
│   │   ├── multiple_choice.py
│   │   ├── perplexity.py
│   │   ├── performance.py
│   │   └── memory.py
│   ├── metrics.py
│   └── report.py
├── board/
│   ├── rknn_eval_agent.py           # Toolkit Lite 板端常驻进程
│   └── runtime_probe/                # 可选 C++ profile/内存工具
├── schemas/
│   ├── request.schema.json
│   └── result.schema.json
└── results/<experiment_id>/
    ├── manifest.json
    ├── predictions.jsonl
    ├── metrics.json
    ├── profile_ops.txt
    ├── profile_mem.txt
    └── report.md
```

## 4. 板端 Agent

### 4.1 生命周期

板端 Agent 启动后只加载一次模型：

1. `RKNN3Lite(llm_mode=True)`；
2. `load_rknn(model_path, weight_path)`；
3. 查询模型 LLM 配置，获取 `vocab_size`、`embedding_dim`、最大上下文等，不在配置中硬编码；
4. 加载 tokenizer 和可选 `embed.bin`；
5. 注册 tokenizer、embedding、output、sampling、result 回调；
6. `init_runtime(target, core_mask, llm_args, llm_callback)`；
7. 创建/复用 Session 执行请求；
8. 测试批次完成后释放 Session 和 Runtime。

`core_mask` 必须与模型构建时使用的 NPU 核数匹配：RK1820/RK1828 可用 `0x1~0xff`，RK3572 仅使用 `0x1`。

### 4.2 请求协议

第一版使用一行一条 JSON 的 stdin/stdout 协议，便于通过 SSH 管道运行，也避免引入 Web 服务依赖：

```json
{"id":"sample-1","op":"generate","prompt":"...","generation":{"max_new_tokens":128,"top_k":1,"temperature":1.0}}
{"id":"sample-2","op":"score","tokens":[151644,872,198,...]}
{"id":"profile-1","op":"benchmark","prompt_tokens":128,"new_tokens":128,"repeat":10}
```

响应：

```json
{
  "id": "sample-1",
  "ok": true,
  "output_text": "...",
  "output_tokens": [123, 456],
  "timing": {
    "prefill_tokens": 32,
    "decode_tokens": 128,
    "ttft_ms": 81.2,
    "prefill_ms": 70.4,
    "decode_ms": 1430.7
  }
}
```

日志必须写入 stderr，stdout 只允许输出协议 JSON，防止破坏结果解析。

### 4.3 Session 隔离

- 独立样本默认 `keep_history=0`，每条样本前清理 KVCache；
- 多轮对话任务在同一 `conversation_id` 内使用 `keep_history=1`；
- 不同样本绝不能复用历史 KVCache；
- 性能评测固定单 Session、单并发；吞吐扩展测试另设 `concurrency` 组，不与单请求性能混合。

## 5. 评估能力

### 5.1 冒烟测试

每个模型先执行以下硬门禁：

- 所有部署文件存在且 SHA-256 与 manifest 一致；
- Runtime、Toolkit Lite、通信服务版本一致；
- 模型能加载和初始化；
- 输入一个短 prompt 能在超时前返回 EOS 或达到 `max_new_tokens`；
- 输出 token 均在词表范围内，无 NaN/Inf logits（若已启用 logits 回调）；
- 相同输入在 greedy 配置下重复两次，token 序列一致；
- Session 释放后板端内存能够回落到容许范围。

冒烟测试失败时不继续跑完整数据集。

### 5.2 生成类任务

Host 侧负责 prompt 模板和指标，Board Agent 只负责 tokenization/generation。首期可接入：

- `lambada`：末词准确率；
- `gsm8k`：答案抽取后的 exact match；
- `ceval` / `mmlu` / `arc`：固定模板生成选项字母；
- 自定义 JSONL：exact match、contains、ROUGE/BLEU 等。

用于精度比较时统一使用 greedy：`top_k=1`、关闭随机采样、固定最大生成长度。任务模板、stop tokens 和答案抽取器必须写入结果 manifest。

### 5.3 选择题优先采用候选项打分

如果 `score` 能力已实现，MMLU/CEval/ARC 应计算每个候选答案的条件对数似然并选择最大者，而不是依赖模型生成单个字母。这样更接近现有 `lm-evaluation-harness` 语义，也能减少输出格式导致的误判。

若第一阶段尚未实现 logits 回调，可先使用 greedy 生成模式，但报告中必须标记：

```json
{"evaluation_mode":"generation_fallback"}
```

不能把该结果与候选项 loglikelihood 结果直接横向比较。

### 5.4 PPL

当前 `eval_ppl.py` 直接操作 PyTorch 层和 CUDA Tensor，无法复用于 RKNN Session，需要单独实现板端 `score` 接口。

推荐利用 LLM Session 的 `output_callback` 取得 logits，并用 `sampling_callback` 执行 teacher forcing：

1. Host 用与板端一致的 tokenizer 生成 token 序列；
2. Board 每一步保留目标 token 的 logits，计算 `log_softmax(logits)[target]`；
3. 自定义 sampling callback 强制返回真实下一个 token，而不是模型预测 token，使 KVCache 始终对应真实上下文；
4. 累计有效 token 的负对数似然；
5. Host 汇总 `ppl = exp(sum_nll / valid_token_count)`。

实现 PPL 前必须用一个很短的固定 token 序列验证：logits 与 target 的位移关系正确、BOS/EOS 是否计分正确、长文本截断与滑窗规则明确。如果当前 Toolkit Lite 暴露的 callback 不足，则在 `board/runtime_probe` 中用 Runtime C Session API 实现 `score`，不能用生成概率近似 PPL。

### 5.5 性能

性能测试至少提供三组输入长度：128、512、1024；输出长度建议固定为 128。每组：

- 模型加载和 Session 初始化不计入推理延迟；
- 预热 3~5 次；
- 正式重复至少 10 次；
- 固定 CPU/NPU 频率并记录实际频率；
- 记录原始样本，不只记录平均值；
- 汇总 p50/p90，并标明是否发生 thermal throttling。

核心指标：

```text
TTFT(ms)       = 首 token 时间 - 请求开始时间
Prefill TPS    = prefill_tokens / prefill_seconds
TPOT(ms/token) = decode_seconds / decode_tokens
Decode TPS     = decode_tokens / decode_seconds
E2E(ms)        = 请求完成时间 - 请求开始时间
```

Toolkit Lite `session_run()` 返回的 token 数与起止时间用于日常回归；发布前再用 Runtime 的 `rknn3_session_query_state()` 或等价接口核验。逐算子分析使用 `rknn3_profile_ops()`，不能用包含 SSH/HTTP 的端到端时间代替 NPU 性能。

### 5.6 内存

分别报告：

- 模型磁盘体积：`.rknn`、`.weight`、`.embed.bin` 等；
- Runtime/NPU：Command、Weight、Internal、KVCache、Total；
- Host/板端进程：Agent RSS 和峰值 RSS；
- 不同 `kvcache_buffer_len`、`kvcache_dtype` 和 Session 数量下的增量。

NPU 内存以 `rknn3_profile_mem()` 或相应 Runtime 查询结果为准。进程 RSS 只作为补充，两者不能相加后当作同一个内存指标。

## 6. 配置设计

建议配置示例：

```yaml
schema_version: 1
experiment_id: qwen2_5_0_5b_w4a16_rk1820

model:
  name: Qwen2.5-0.5B-Instruct
  type: llm
  rknn: /opt/models/qwen2_5_0_5b/model.rknn
  weight: /opt/models/qwen2_5_0_5b/model.weight
  tokenizer: /opt/models/qwen2_5_0_5b/model.tokenizer.gguf
  embed: /opt/models/qwen2_5_0_5b/model.embed.bin

device:
  platform: rk1820
  core_mask: 0xff
  transport: ssh
  ssh_host: rk3588-board
  workdir: /opt/ptq-bench

session:
  max_context_len: 1024
  max_new_tokens: 128
  keep_history: false
  sampling:
    top_k: 1
    top_p: 1.0
    temperature: 1.0  # top_k=1 时仍为确定性生成，并避免部分 Runtime 拒绝 0
    repeat_penalty: 1.0

evaluation:
  smoke: true
  tasks: [lambada, ceval-valid]
  perplexity:
    enabled: false
    datasets: [wikitext2]
    sequence_length: 1024
  benchmark:
    prompt_lengths: [128, 512, 1024]
    new_tokens: 128
    warmup: 5
    repeat: 10
  memory: true
```

配置加载时必须做语义校验：

- RK3572 的 `core_mask` 必须为 `0x1`；
- `prompt_length + new_tokens` 不得超过模型最大上下文；
- 非 Tie Word 模型必须配置 `embed`；
- PPL/候选打分要求后端声明 `supports_logits=true`；
- 多轮任务与独立样本任务不能共享同一 Session 状态；
- 性能模式必须禁止随机采样和并发任务干扰。

## 7. 结果格式

`metrics.json` 不使用只面向人阅读的 stdout 文本，建议结构如下：

```json
{
  "schema_version": 1,
  "experiment_id": "qwen2_5_0_5b_w4a16_rk1820",
  "status": "passed",
  "environment": {
    "platform": "rk1820",
    "core_mask": "0xff",
    "runtime_version": "...",
    "toolkit_lite_version": "..."
  },
  "accuracy": {
    "ceval-valid": {"metric": "acc", "value": 0.0, "mode": "loglikelihood"}
  },
  "performance": {
    "prompt_128_output_128": {
      "ttft_ms_p50": 0.0,
      "decode_tps_p50": 0.0,
      "e2e_ms_p90": 0.0
    }
  },
  "memory_mb": {
    "weight": 0.0,
    "internal": 0.0,
    "kvcache": 0.0,
    "total": 0.0,
    "process_peak_rss": 0.0
  }
}
```

每条预测单独写入 `predictions.jsonl`，包含 sample id、prompt 哈希、原始输出、解析后答案、耗时和错误；这样失败样本可重跑，也便于核查指标。

## 8. 执行入口

建议 Host 使用统一命令：

```bash
python run_rknn_eval.py preflight --config configs/rknn/qwen2_5_0_5b_rk1820.yaml
python run_rknn_eval.py smoke     --config configs/rknn/qwen2_5_0_5b_rk1820.yaml
python run_rknn_eval.py accuracy  --config configs/rknn/qwen2_5_0_5b_rk1820.yaml
python run_rknn_eval.py benchmark --config configs/rknn/qwen2_5_0_5b_rk1820.yaml
python run_rknn_eval.py all       --config configs/rknn/qwen2_5_0_5b_rk1820.yaml
```

`all` 的固定顺序为：preflight -> smoke -> accuracy -> benchmark -> memory -> report。任一硬门禁失败即停止，避免在错误模型或错误核心配置上浪费长时间评测。

## 9. 实施阶段

### 阶段一：板端闭环

- 支持单个纯文本 LLM；
- Toolkit Lite 加载、greedy 生成、JSONL 协议；
- 冒烟、生成任务、TTFT/Prefill/Decode 性能；
- `manifest.json`、`predictions.jsonl`、`metrics.json`。

验收标准：同一 greedy 请求重复结果一致；100 条样本连续执行无 Session 状态串扰和明显内存增长。

### 阶段二：标准评测

- 接入 `lm-evaluation-harness` 自定义 RKNN backend；
- 实现 logits/output callback 和候选项打分；
- 实现 Wikitext2 PPL；
- 完善异常重试、断点续跑和样本级错误记录。

验收标准：短序列 teacher-forcing 单元测试通过；PPL 可重复；生成模式与 loglikelihood 模式在报告中严格区分。

### 阶段三：硬件剖析与扩展

- Runtime C++ probe；
- 算子 profile、NPU/进程内存、频率和温度采集；
- 多 Session/并发吞吐测试；
- 再按需求扩展 MLLM。MLLM 需要分别加载 vision/audio 与 LLM 模型，并将编码器 embedding 作为 Session 多模态输入，不能直接复用纯文本输入适配器。

## 10. 风险与约束

1. RKNN3 Toolkit、Toolkit Lite、Runtime 和通信服务必须保持版本一致；版本不一致时结果不可信。
2. 模型产物是组合件，`.rknn` 与 `.weight`（以及 tokenizer/embed）混用版本可能加载失败或产生异常结果，必须用 manifest 哈希绑定。
3. `rkllm3-server` 的 HTTP 延迟不能代表 NPU 推理性能。
4. 只生成答案但不暴露 logits 时，无法严格实现 PPL 和多数 `lm-eval` 的 loglikelihood 任务。
5. KVCache 长度、类型、核心数均在模型构建阶段影响产物；板端评估只能测已有产物，不能在评估脚本里任意改变这些条件。
6. 当前工程中的 Mamba、LLaVA/VILA 等模型不能仅凭文件名判定可在 RKNN3 上运行，必须已有与 Runtime 匹配的 RKNN 产物和对应输入适配器后才纳入板端评测。

## 11. 推荐的最小落地范围

第一版先完成：

- 目标平台选一种（RK182X 或 RK3572）；
- 一个已验证的纯文本 LLM RKNN 模型；
- Toolkit Lite Board Agent；
- `smoke + greedy accuracy + TTFT/Decode TPS + memory`；
- Host 侧 JSONL 数据集和统一 JSON 报告。

PPL、完整 `lm-evaluation-harness` 和 C++ profile 放到第二阶段。这样能够先形成稳定的板端评估闭环，同时保留后续严格 loglikelihood 评测所需的协议和扩展点。
