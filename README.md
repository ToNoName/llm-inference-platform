# LLM Interfence Gateway

一个支持多引擎(vllm + llama.cpp) 的生产级大模型推理网关，兼容OpenAI API格式，支持流式输出，一键Docker部署

## 架构
Client → Nginx (:80) → FastAPI Gateway (:8000) → vLLM (:8001) → llama.cpp (:8002)
![架构图](gateway_drawio.png)

## 特性
- ✅ **OpenAI 兼容**：标准 `/v1/chat/completions` 接口，可直接替换 OpenAI SDK 的 base_url
- ✅ **双后端智能路由**：根据模型名自动分发到 vLLM（高性能）或 llama.cpp（轻量CPU推理）
- ✅ **流式输出**：支持 SSE (Server-Sent Events) 打字机效果
- ✅ **高并发保护**：Pydantic 请求校验 + httpx 连接池限制 + 分层错误处理
- ✅ **容器化部署**：Docker + Docker Compose 一键启动完整推理集群
- ✅ **健康检查**：Nginx 反向代理 + 网关健康监控

## 快速开始

### 前置要求
- Docker & Docker Compose
- NVIDIA GPU + nvidia-container-toolkit（如需 vLLM）

### 一键启动

```bash
git clone https://github.com/ToNoName/llm-inference-platform.git
cd llm-inference-platform
docker-compose up -d
```

### 非流式请求
```bash
curl -X POST http://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

### 流式请求
```bash
curl -N -X POST http://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "max_tokens": 50
  }'
```
# 项目结构
```bash
llm-inference-platform/
├── src/
│   └── gateway/
│       ├── __init__.py
│       └── gateway.py          # FastAPI 网关核心代码
├── docker/
│   ├── Dockerfile              # 网关镜像
│   └── nginx/
│       └── nginx.conf          # Nginx 反向代理配置
├── docker-compose.yml          # 全栈编排
├── requirements.txt
└── README.md
```

# 技术栈
```bash
FastAPI + httpx：异步高并发网关

vLLM：主力推理引擎（PagedAttention, Continuous Batching）

llama.cpp：轻量级 CPU/GPU 混合推理

Nginx：反向代理 + SSE 流式支持

Docker Compose：一键部署
```


