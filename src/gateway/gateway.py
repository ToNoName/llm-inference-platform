# gateway.py
########################################
# 20260525 - 创建基础版本 支持 OpenAI 兼容风格的请求转发到 vLLM 后端, 并处理流式响应.
########################################
import httpx
from fastapi import FastAPI, Request,HTTPException
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel, Field
from typing import List, Optional
import os

app = FastAPI(title="Inference Gateway")

VLLM_BACKEND_URL = os.getenv("VLLM_BACKEND_URL", "http://host.docker.internal:8001/v1/chat/completions")
LLAMA_BACKEND_URL = os.getenv("LLAMA_BACKEND_URL", "http://host.docker.internal:8002/v1/chat/completions")

HTTPX_LIMITS = httpx.Limits(max_keepalive_connections=10, max_connections=20)
HTTPX_TIMEOUT = 120.0

class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色，如 'user' 或 'assistant'")
    content: str = Field(..., description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default = "qwen", description="模型名称")
    messages: List[ChatMessage] = Field(..., description = "聊天消息列表")
    temperature: Optional[float] = Field(default = 0.7, ge = 0.0, le = 2.0, description="生成文本的随机程度,范围0-2")
    max_tokens: Optional[int] = Field(default = 256, ge = 2, le = 4096, description="生成文本的最大令牌数")
    stream: Optional[bool] = Field(default=False, description="是否启用流式响应")

async def stream_forwarder(payload: dict, backend_url: str):
    async with httpx.AsyncClient(limits=HTTPX_LIMITS, timeout=HTTPX_TIMEOUT) as client:
        try:
            async with client.stream("POST", backend_url, json=payload) as response:
                # 检查后端响应状态
                if response.status_code != 200:
                    err = {'error': 'vLLM backend error', 'status_code': response.status_code}
                    yield f"data: {json.dumps(err,ensure_ascii=False)}\n\n"
                    return

                # 逐块转发 vLLM 的响应
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.RequestError as e:
            err = {'error': 'Request to vLLM backend failed', 'details': str(e)}
            yield f"data: {json.dumps(err,ensure_ascii=False)}\n\n"


#OpenAI 兼容风格网关入口
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 1. 读取原始请求体
    payload = request.model_dump()  # 将 Pydantic 模型转换为字典
    is_stream = payload.get("stream", False) # 获取是否启用流式响应的标志

    # 2. 根据模型名称选择后端 URL
    if "llama" in request.model.lower():
        backend = LLAMA_BACKEND_URL
    else:
        payload["model"] = "qwen2.5-3b"
        backend = VLLM_BACKEND_URL

    print(f"[Gateway] Routing model '{request.model}' to {backend}")

    try:
        if is_stream:
            # 流式请求：逐块转发 vLLM 的响应
            return StreamingResponse(stream_forwarder(payload,backend), media_type="text/event-stream")
        else:
            # 用异步 HTTP 客户端转发到 vLLM
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT, limits=HTTPX_LIMITS) as client:
                resp = await client.post(backend, json=payload)
                resp.raise_for_status()  # 如果 vLLM 返回错误状态码，抛出异常
                return resp.json()
    
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"后端服务异常: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "inference-gateway"}
