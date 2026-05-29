from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import List, Optional
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI( title="LLM推理网关", description="一个基于FastAPI的LLM推理网关示例", version="1.0.0" )

# ----------- 数据模型定义 -----------

class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色，如 'user' 或 'assistant'")
    content: str = Field(..., description="消息内容")

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge = 1, le=4096) # ge = 大于等于，le = 小于等于
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)

class ChatResponse(BaseModel):
    content: str
    tokens_used: int

# ----------- API端点定义 -----------
@app.get("/")
async def root():
    return {"message": "欢迎来到LLM推理网关!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu" : "RTX 5060" }

@app.post("/v1/chat/completions", response_model=ChatResponse) # /v1/chat/completions网址路径 response_model=ChatResponse：返回的数据必须长这样
async def chat_completions(request: ChatRequest):
    # 这里我们简单地将用户消息拼接成一个响应，实际应用中应该调用LLM进行生成
    user_message = " ".join([msg.content for msg in request.messages if msg.role == "user"])  #" ".join()：把列表中的字符串用空格连接成一个字符串
    assistant_response = f"你说的是 '{user_message}' 。 这是模拟的回复。"
    tokens_used = len(assistant_response.split()) # split()：把句子按空格切成列表
    
    return ChatResponse(content=assistant_response, tokens_used=tokens_used)

@app.post("/v1/chat/async-demo")
async def async_chat(request: ChatRequest):
    # 模拟一个异步生成过程，实际应用中应该调用LLM进行生成
    await asyncio.sleep(1)  # 模拟生成延迟
    assistant_response = f"这是一个异步模拟回复，异步处理完成了用户消息。"
    return ChatResponse(content=assistant_response, tokens_used=len(assistant_response.split()))



#-------------- streaming response示例 （SSE） --------------
async def generate_stream(user_msg: str,max_tokens: int):
    """一个模拟的流式生成器，异步执行"""
    tokens = ["这是", "一个", "模拟的", "LLM", "生成的", "文本。"] * (max_tokens // 6)  # 模拟生成的tokens数量
    for token in tokens:
        await asyncio.sleep(0.01)  # 模拟每个token生成的时间
        chunk = {"choices": [{"delta": {"content": token} ,"index": 0}]}
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"  # SSE格式

    yield "data: [DONE]\n\n"  # SSE结束标志

@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    user_message = " ".join([msg.content for msg in request.messages if msg.role == "user"])
    return StreamingResponse(generate_stream(user_message, request.max_tokens), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive","X-Accel-Buffering": "no"})


