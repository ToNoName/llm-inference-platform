# 理论知识库

面试前 30 分钟快速翻阅。

---

## 文件索引

| 文件 | 一句话 | 关键词 |
|------|--------|--------|
| [kv-cache-paged-attention.md](kv-cache-paged-attention.md) | KV Cache → PagedAttention → Continuous Batching → 带宽瓶颈 | block pool, block_table, gpu_util, max_num_seqs |
| [inference-metrics.md](inference-metrics.md) | TTFT / TPOT / P50 / P90 / P99 / SLO 定义与使用 | 百分位, 抖动, 生产监控 |
| [quantization.md](quantization.md) | GPTQ / AWQ / GGUF 原理与选型 | INT4, Q4_K_M, 显存节省 |
| [engine-comparison.md](engine-comparison.md) | vLLM vs llama.cpp vs SGLang | PagedAttention, 连续 KV, 使用场景 |
| [gateway-design.md](gateway-design.md) | 网关路由 / fallback / trace_id / 连接池 | input_tokens 估算, 502/504/500, AsyncClient |

---

## 使用方式

- 每周新增理论直接追加到对应文件末尾（`---` 分隔）
- 面试前看 README 索引 → 哪个概念想不起来就点进去读 2 分钟
- 每篇 300-500 字，白话 + 类比 + 一张 ASCII 图
