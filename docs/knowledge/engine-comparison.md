# 推理引擎对比：vLLM vs llama.cpp vs SGLang

> vLLM 长于高并发在线服务，llama.cpp 强于单用户低延迟+灵活部署，SGLang 面向结构化生成。选引擎本质是选 KV cache 管理策略和调度哲学。

---

## 1. 核心差异

| | vLLM | llama.cpp | SGLang |
|---|------|----------|--------|
| KV Cache | PagedAttention (block pool) | 连续分配 (独占空间) | RadixAttention (前缀共享) |
| 调度 | Continuous Batching | 无调度（顺序执行） | Structured Generation |
| 量化 | GPTQ / AWQ 原生 | GGUF（极灵活） | 复用 vLLM 生态 |
| 硬件 | GPU 优先 | CPU/GPU/Metal 通吃 | GPU |
| 优势场景 | 高并发在线 API | 低显存 / 边缘 / 单用户 | 结构化 JSON 输出 |

---

## 2. 并发扩展对比

```
vLLM (PagedAttention):     c=1→30  TPOT +41%
llama.cpp (连续 KV):        c=1→16  TPOT +9.4×
```

vLLM 的 block pool 让多个请求共享显存物理空间，减少带宽争抢。llama.cpp 每个请求独占一块，并发越高 KV 读取总量越大。

---

## 3. 场景选型

| 场景 | 选引擎 | 原因 |
|------|--------|------|
| 短请求 + 低并发 | llama.cpp | 低延迟，低显存 |
| 长请求 + 高并发 | vLLM + GPTQ | PagedAttention block 共享 |
| 受限显存 <8GB | llama.cpp GGUF | 灵活量化 + CPU fallback |
| 生产 GPU 集群 | vLLM | Continuous Batching + 高吞吐 |
| 需要 JSON 输出 | SGLang | RadixAttention + 结构化生成 |

---

## 4. 面试一句

> "vLLM 的 PagedAttention 把 KV cache 分块共享，高并发下带宽效率远超 llama.cpp 的连续分配。但 llama.cpp 的 GGUF 量化 + CPU 兼容性让它成为受限硬件的标配。没有银弹，看场景选引擎。"
