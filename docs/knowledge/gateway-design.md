# 推理 API 网关设计

> 双后端（vLLM + llama.cpp）自动分流，input_tokens 估算路由，<0.02ms 转发开销，502/504/500 统一错误码 + trace_id。

---

## 1. 架构

```
Client → Nginx :8000 → Gateway (FastAPI)
                          ├─ vLLM :8001      (input > 512)
                          └─ llama.cpp :8002  (input ≤ 512)
```

---

## 2. 路由策略

**Text-based token 估算**（不加载 tokenizer，网关轻量）：

```python
def estimate_tokens(messages):
    text = " ".join(msg.content for msg in messages)
    cn_ratio = sum(1 for c in text if "\u4e00" <= c <= "\u9fff") / max(len(text), 1)
    if cn_ratio > 0.3:
        return int(len(text) * 0.5)   # 中文 ~2 chars/token
    return int(len(text) * 0.25)      # 英文 ~4 chars/token
```

**决策**：`estimate_tokens > 512 → vLLM（PagedAttention）, else → llama.cpp（低延迟）`

**手动切换**：`POST /admin/switch-backend {"mode": "vllm"|"llama"|"auto"}`

---

## 3. Fallback & 错误码

| 错误码 | 原因 | 响应 |
|--------|------|------|
| 502 | 后端不可达 | `{"error":"backend_unavailable","trace_id":"xxx"}` |
| 504 | 推理超时 (>60s) | `{"error":"backend_timeout","trace_id":"xxx"}` |
| 500 | 网关内部异常 | `{"error":"internal_error","trace_id":"xxx"}` |

用户报 trace_id，运维直接查日志定位问题。

---

## 4. 连接池

```python
app.state.http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(REQUEST_TIMEOUT),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
)
```

复用连接 → 避免每请求 TCP/TLS 握手（省 2-5ms）。lifespan 管理生命周期。

---

## 5. 开销

```
Gateway overhead = e2e_latency_ms - backend_latency_ms
                 < 0.02ms (c=1), < 0.12ms (c=30)
                 < 2% 总延迟
```

---

## 6. 面试一句

> "双后端网关根据 input_tokens 自动分流：≤512 走 llama.cpp 低延迟短请求，>512 走 vLLM PagedAttention 长上下文。网关用 text-based token 估算（误差 ±20%，不加载 tokenizer），AsyncClient 连接池复用，转发开销 <0.02ms。统一错误码 502/504/500 + trace_id 支持生产排查。"
