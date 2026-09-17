# Qwen3.5-4B RKNN 板端评估结果

## 测试信息

- 测试日期：2026-09-16
- 目标板：RK3588 + RK1828
- 运行时：RKNN3 1.1.0
- 数据集：WikiText-2 raw test
- 模型：`Qwen3.5-4B-rknn-official`、`Qwen3.5-4B-Q2N-W4A16-G32`

## WikiText-2 PPL

数据准备与作者 `eval_ppl.py` 的口径对齐：拼接 WikiText-2 raw test 文本，分为
145 个连续、不重叠的 2048-token 块。每块对位置 1 到 2047 计分，因此总预测
token 数为 `145 × 2047 = 296,815`。

最终结果使用全局 NLL 聚合，而不是对每段 PPL 做算术平均：

```text
PPL = exp(sum(NLL) / sum(scored_tokens))
```

| 板端模型 | NLL 总和 | Mean NLL | PPL | 端到端评估速度 | 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-rknn-official | 680,208.978736 | 2.291693407 | 9.891674141 | 27.9455 tok/s | 2:57:01 |
| Qwen3.5-4B-Q2N-W4A16-G32 | 671,560.063208 | 2.262554329 | **9.607598820** | 28.0497 tok/s | 2:56:22 |

Q2N 相比 RKNN Official 的 PPL 绝对降低 `0.284075321`，相对降低约 `2.87%`。
作者公布的未部署 Q2N PPL 为 `9.507355`；本次板端结果相对增加约 `1.05%`。
未部署结果和板端结果涉及不同执行后端，差值只能用于部署回归参考。

## Prefill 1024 性能

性能测试与 PPL 解耦，配置为 1024-token Prefill、128-token Decode、预热 3 次、
正式重复 10 次。下表采用 p50。

| 板端模型 | TTFT | Prefill 吞吐 | 模型 Decode 吞吐估计 |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B-rknn-official | 1.360736 s | 752.4632 tok/s | 41.4194 tok/s |
| Qwen3.5-4B-Q2N-W4A16-G32 | 1.360881 s | 752.2698 tok/s | 41.2831 tok/s |

PPL 评估中的约 28 tok/s 包含 CPU 全词表 NLL 归约开销，不应与性能探针给出的
纯模型 Decode 吞吐直接混用。

## 内存

RK3588 + RK1828 采用统一内存体系，这里报告 RKNN 运行时分配和 Host 进程 RSS，
不使用 CUDA“显存”口径。两个模型报告的分配相同。

| 项目 | 字节 | 换算值 |
| --- | ---: | ---: |
| RKNN 总分配 | 3,026,887,680 | 2.819 GiB |
| 权重分配 | 2,647,438,336 | 2.466 GiB |
| KV Cache | 297,697,280 | 283.9 MiB |
| Host 峰值 RSS | 159,891,456 | 152.5 MiB |

## 上下文能力

本轮以 `--max-context-len 4096` 成功初始化 Session，并完成 2048-token PPL 和
1024-token Prefill + 128-token Decode 测试。因此 4096 是当前已验证上下文长度，
不是通过 4096、8192 等档位边界扫描得到的最大上下文上限。

## 结论

- Q2N 在板端的 WikiText-2 PPL 优于 RKNN Official，降低约 2.87%。
- 两个模型的 Prefill、Decode、TTFT 和运行时分配基本一致。
- Q2N 板端 PPL 与作者未部署结果相差约 1.05%，后续可作为转换链路回归基线。
- C4 PPL 和最大上下文边界扫描尚未包含在本轮正式结果中。
