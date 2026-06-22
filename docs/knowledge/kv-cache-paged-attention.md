# KV Cache → PagedAttention → Continuous Batching

> KV Cache 是加速自回归生成的缓存。llama.cpp 用连续分配，vLLM 用分页管理。后者在高并发下带宽效率远超前者。

---

## 1. KV Cache：为什么需要它

### 自回归生成过程

```
Input: "今天天气" (4 tokens)

Prefill:
  一次性算出 "今天天气" 这 4 个 token 的 K、V，存入缓存

Decode (逐个生成):
  Step 1: Q_new 和 [K1..K4] [V1..V4] 做 attention → 生成 token 5 "不错"
          → K5, V5 追加到缓存
  
  Step 2: Q_new 和 [K1..K5] [V1..V5] 做 attention → 生成 token 6 "，"
          → K6, V6 追加到缓存

  ... 重复直到 <eos>
```

**KV Cache 的作用**：缓存历史上文 token 的 K、V 向量，避免每步都从头重算所有 token 的 attention。

### 一个 token 占多少 KV？

```
per_token_bytes = num_layers × num_kv_heads × head_dim × 2(K+V) × dtype

Qwen2.5-7B: 28 × 4 × 128 × 2 × 2 = 57,344 bytes ≈ 56 KB/token
```

---

## 2. llama.cpp：连续分配

```
启动时一次性分配一整块连续显存：

┌─────────────────────────────────────────────────────────┐
│  Req 0 KV (预分 27K token) │ Req 1 KV │ Req 2 KV │ ... │
│  [##########_____________]  │  [...]   │  [...]   │     │
│   └─ 实际 200 token ─┘      │           │          │     │
│   └─ 预分配 27000 token ───┘           │          │     │
└─────────────────────────────────────────────────────────┘
```

**特点**：
- 每个请求独占一块连续空间，启动时就分配好最大容量
- 未使用的空间闲置，别人不能用
- 读取时每请求独立寻址，16 并发 = 16 次分散读取
- 优势：管理简单，无块表查表开销 → 极低延迟（单请求场景）

---

## 3. vLLM PagedAttention：Block Pool

### Block 是什么

```
Block Pool（公共池）：
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ ... 每个 16 token
└────┴────┴────┴────┴────┴────┴────┴────┘

block_table（逻辑映射）：
  Req 0: B0 → B3 → B7   (48 token)
  Req 1: B1 → B5         (32 token)
  Req 2: B4 → B6 → B2   (48 token)  ← 注意：物理地址可以乱序！
```

### 生命周期

```
1. 启动：num_gpu_blocks = (gpu_util × total_vram - model_size) / block_size_bytes
2. 请求到达：scheduler 从 pool 取空闲 block 分配给 sequence
3. Prefill：计算 prompt 的 KV，占满第一个 block 后申请下一个
4. Decode：每步生成新 token，block 满了就申请新 block
5. 完成/超时：block 全部归还 pool，其他人可用
```

### 优势

- 物理地址可以离散，逻辑地址由 block_table 维护
- 按需分配，用完归还，无闲置浪费
- 多个请求的 block 物理相邻时，GPU 可以合并读取 → 带宽效率高

---

## 4. Continuous Batching

vLLM 的 scheduler 不等待一个请求全部完成再开始下一个。

```
Scheduler 循环（每 ~6ms 一次 forward）：

  running = [所有 block 够用的 waiting 请求]
  if len(running) >= max_num_seqs:
      超出的放回 waiting
  
  for seq in running:
      做一次 forward（1 个 token）
      if seq 生成了 <eos>: done
      if seq block 不够: 申请新 block，没有 → 放回 waiting，preempt
```

**关键**：请求在任意 step 可以加入或退出 batch。不像 Static Batching 必须等整个 batch 全部完成。

---

## 5. GPU 显存带宽：Concurrency 的最终瓶颈

### 每步 Decode 的数据量

```
数据量 = 模型权重 + 所有并发请求的 KV Cache 总和

c=1  (128in):  3.5GB(权重) + 7MB(KV)    ≈ 3.5GB
c=16 (128in):  3.5GB(权重) + 112MB(KV)  ≈ 3.6GB  ← KV 占比 3%
c=16 (1024in): 3.5GB(权重) + 896MB(KV)  ≈ 4.4GB  ← KV 占比 20%
```

### 为什么 llama.cpp 更容易瓶颈

```
llama.cpp: 每请求独立一块 KV → 16 次分散读取 → 带宽利用效率低
vLLM:     block pool 共享 → 物理相邻 block 可合并读取 → 带宽利用效率高
```

| 引擎 | c=1→16 TPOT 增幅 | c=1→30 TPOT 增幅 |
|------|-----------------|-----------------|
| llama.cpp Q4_K_M | **+9.4×** | — |
| vLLM GPTQ-Int4 | +25% | **+41%** |

---

## 6. 面试一句话

> "vLLM 的 PagedAttention 把 KV cache 切成固定大小 block，扔进公共池子按需分配。请求完成 block 归还，多个请求共享物理显存，GPU 可以合并读取相邻 block。这比 llama.cpp 的连续独占分配在高并发场景下带宽效率高得多。**这不是技术优劣之分，是场景选择：单用户低延迟选 llama.cpp，高并发在线服务选 vLLM。**"

### 速记公式

```
gpu_util → block_pool_size → max_concurrent
per_token ≈ 56KB (7B, FP16)
单请求 TPOT ≠ 跟 gpu_util 无关
并发 TPOT = 带宽争抢程度
```
