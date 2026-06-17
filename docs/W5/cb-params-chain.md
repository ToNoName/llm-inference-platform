# vLLM Continuous Batching 参数链分析

> Qwen2.5-7B + GPTQ-Int4 | RTX 4090D 24GB | vLLM 0.22.1

---

## 1. 关键概念

| 术语 | 定义 |
|------|------|
| **TTFT** (Time To First Token) | 请求发出到首 token 出现，包含 prefill + 网络 |
| **TPOT** (Time Per Output Token) | 生成阶段每个 output token 平均耗时 |
| **Attention**（注意力机制） | 计算 token 间相关性权重，融合上下文。Decode 阶段因果遮罩（只看前面），Encoder 阶段双向可见 |
| **KV Cache** | 缓存 Prefill 阶段已算出的历史 K、V 向量。Decode 时只增量计算新 token 的 QKV，避免重复完整 attention 计算 |
| **OOM** | Out of Memory，显存溢出 |
| **quant** | 量化级别 |
| **concurrency** | 并发请求数 |

---

## 2. gpu_util 不影响单请求延迟

**实验对比**（exp1 vs exp4，c=1）：

| exp | gpu_util | TTFT_P50 | TPOT_P50 | VRAM |
|-----|----------|---------|---------|------|
| exp1 | 0.3 | 798ms | 6.23ms | 7007 MiB |
| exp4 | 0.9 | 798ms | 6.23ms | 22183 MiB |

**结论**：gpu_util 从 0.3 调到 0.9，TTFT 和 TPOT 完全不变。

**原因**：

```
单请求 TPOT = 模型计算时间 + 显存带宽时间
            ≠ KV cache 预分配大小

显存带宽时间是瓶颈——每一步需要：
1. 从显存读取所有权重
2. 读取历史 KV Cache 参与 Attention 计算
3. 写入新的 KV Cache
计算时间极短，带宽等数据才是大头。
```

单请求只消耗 1 条 sequence 的 KV cache。预分配再多 block，用不上就是白分配。

---

## 3. PagedAttention 并发扩展

**实验对比**（GPTQ-Int4 baseline，gpu_util=0.9）：

| 并发 | TPOT_P50 | TPOT_P99 | 相对 c=1 |
|------|---------|---------|---------|
| 1 | 6.18ms | 6.22ms | — |
| 4 | 6.45ms | 6.59ms | +4.4% |
| 8 | 6.71ms | 6.94ms | +8.6% |
| 16 | 7.71ms | 8.56ms | +24.8% |
| **30** | **8.69ms** | **9.66ms** | **+40.6%** |

**结论**：30 倍并发，TPOT 只增长 40%。

**原因**：PagedAttention 的 block 共享机制让多条 sequence 的 KV cache 高效复用显存带宽。如果传统方案每条 sequence 独占一块连续显存，c=30 时要么 OOM 要么延迟 ×30。

---

## 4. gpu_util 控制并发容量上限

### 4.1 VRAM 实验数据

| exp | gpu_util | VRAM | 模型权重 | 可用 KV Cache |
|-----|----------|------|---------|-------------|
| exp1 | 0.3 | 7007 MiB | ~6.5 GB | **~500 MB** |
| exp2 | 0.5 | 12551 MiB | ~6.5 GB | ~6 GB |
| exp3 | 0.7 | 17367 MiB | ~6.5 GB | ~10.5 GB |
| exp4 | 0.9 | 22183 MiB | ~6.5 GB | **~15 GB** |

### 4.2 KV Cache 计算

**每个 token 的 KV 占用**：

```
per_token_bytes = num_layers × num_kv_heads × head_dim × 2(K+V) × dtype
```

**每个请求的 KV 占用**：

```
request_kv_mb = sequence_length × per_token_bytes
```

**最大并发数**：

```
max_concurrent = available_kv_cache / request_kv_mb
```

### 4.3 Qwen2.5-7B 实例

| 参数 | 值 | 来源 |
|------|-----|------|
| num_layers | 28 | `config.json` → `num_hidden_layers` |
| num_kv_heads | 4 | `config.json` → `num_key_value_heads`（GQA） |
| head_dim | 128 | `config.json` → `hidden_size / num_attention_heads` |
| dtype | 2 bytes | FP16，KV cache 不量化 |

```
per_token_bytes = 28 × 4 × 128 × 2 × 2 = 57,344 bytes ≈ 56 KB/token
```

| 请求长度 | KV Cache 占用 |
|---------|-------------|
| 128 token | 128 × 56KB = **7 MB** |
| 512 token | 512 × 56KB = **29 MB** |
| 1024 token | 1024 × 56KB = **56 MB** |
| 4096 token | 4096 × 56KB = **229 MB** |

### 4.4 不同 gpu_util 的并发容量

| gpu_util | 可用 KV Cache | 128 token 并发 | 1024 token 并发 | 4096 token 并发 |
|----------|-------------|--------------|---------------|---------------|
| 0.3 | ~500 MB | ~71 个 | ~8 个 | ~2 个 |
| 0.5 | ~6 GB | ~850 个 | ~107 个 | ~26 个 |
| 0.9 | ~15 GB | ~2000 个 | ~264 个 | ~65 个 |

**关键发现**：gpu_util 对短请求影响不大（0.3 也能撑 71 个并发），但对长请求影响极大（0.3 只能 2 个 vs 0.9 能 65 个）。**这就是路由策略中 input_tokens > 512 → vLLM 的底层原因。**

---

## 5. max_num_seqs 控制调度器上限

### 5.1 定义

max_num_seqs 是调度器的硬上限：每轮 forward 最多让多少个请求参与计算。**它控制的是调度行为，不控制显存分配。**

### 5.2 实验验证

| exp | gpu_util | max_seqs | VRAM | 差异 |
|-----|----------|----------|------|------|
| exp4 | 0.9 | 128 | 22183 MiB | — |
| exp5 | 0.9 | 8 | 21109 MiB | -5% |
| exp6 | 0.9 | 32 | 21191 MiB | -4% |

max_seqs 从 8 变到 128，VRAM 只差 5%。**证明它不控制显存分配，只约束调度器。**

### 5.3 与 gpu_util 的关系

```
实际并发数 = min(max_num_seqs, 显存能装几个)
```

| 场景 | 结果 |
|------|------|
| max_seqs=128，显存只够 10 个 | 实际跑 10 个（显存瓶颈） |
| max_seqs=8，显存够 100 个 | 实际跑 8 个（调度器瓶颈） |
| max_seqs=128，显存够 100 个 | 全跑（显存瓶颈） |

### 5.4 gpu_util=0.9 内部流程

```
1. 检测总显存 24564 MiB
2. 预留模型权重 ~6.5GB + CUDA graph ~1GB
3. 剩余 ~15GB 全部切成 block（每个 ~10MB, 16 token）
4. ~1500 个 block 堆入 block_pool
5. 请求到达 → scheduler 从 pool 取 block 分配给 sequence
6. 请求完成 → block 归还 pool
7. 并发上限 = min(max_seqs, pool 中可分配的 block 数)
```

### 5.5 生产建议

| max_seqs | 场景 |
|----------|------|
| 8–32 | 低并发，追求低延迟（batch 小 → 每请求算得快） |
| 64–128 | 高并发，追求吞吐（batch 大 → GPU 利用率高） |
| >256 | 一般不设（显存物理上限低于调度器上限） |

---

## 6. 总结

```
gpu_util    = 锅的大小（决定 KV cache 池子能装多少 block）
max_num_seqs = 菜单上限（最多同时做几道菜）
实际并发     = min(菜单上限, 锅能装几道)
单请求速度   = 跟锅大小无关，只跟菜（模型）复杂度有关
```

| 参数 | 控制对象 | 改显存？ | 影响单请求？ | 核心作用 |
|------|---------|---------|------------|---------|
| gpu_memory_utilization | block pool 大小 | 是 | 否 | 并发容量 |
| max_num_seqs | scheduler 上限 | 否 | 否 | 吞吐上限 |
| max_model_len | block 大小 × 数量 | 略 | 略 | 支撑长上下文 |

---

## 7. 内容精简

**Q: gpu_util 调到 0.9 和 0.3 有什么区别？**

> c=1 没区别。c=30 时：0.3 的 KV cache 池只有 500MB，长请求可能只能跑 2 个；0.9 的池有 15GB，可以跑几十个。gpu_util 决定的是"能同时服务多少请求"，不是"单个请求跑多快"。

**Q: max_seqs 调到 128 会多占显存吗？**

> 不会。max_seqs 只约束调度器每回合最多取多少请求参与计算，不改变 block pool 大小。W4 实验证明 max_seqs 从 8 到 128，VRAM 只差 5%。

**Q: 怎么确定 gpu_util 的最佳值？**

> 先设 0.9，不 OOM 就保持。OOM 则降到 0.85/0.8。生产环境留 1-2GB 给 CUDA graph 波动。推公式：per_token ≈ 56KB，按最长请求长度算并发容量是否满足业务需求。

**Q: PagedAttention 和传统 KV Cache 的本质区别？**

> 传统：每个请求预分配一整块连续显存，碎片化严重，并发需求估算不准就 OOM。PagedAttention：切成固定大小 block，所有请求共享 block pool，逻辑连续但物理可以不连续，按需分配归还。block_table 维护映射。
