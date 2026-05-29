from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI(title = "My First API")

@app.get("/")  #访问 http://127.0.0.1:8000  , 
async def root():
    return {"message": "Hello World!"}
#curl http://localhost:8000

@app.get("/non")  #访问 http://127.0.0.1:8000/non/
async def non():
    return {"message": "This is a non-root endpoint."}
# curl http://localhost:8000/non


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 100


@app.post("/chat")
async def chat(request: ChatRequest):
    response_message = f"接收到的消息为: { request.message }，最大令牌数为: { request.max_tokens }"
    return {"response": response_message}

# curl -N -X POST http://localhost:8000/chat  -H "Content-Type: application/json" -d '{ "message": "my first req", "max_tokens": 30 }'

async def token_generator(prompt : str):
    for i in range(5):
        yield f"data token_{ i } \n"
        await asyncio.sleep(0.5)
    yield "data [DONE] \n\n"

@app.post("/chat/stream")
async def chat_stream(req : ChatRequest):
    generator = token_generator(req.message)
    return StreamingResponse(generator, media_type="text/event-stream")

# curl -N -X POST http://localhost:8000/chat/stream  -H "Content-Type: application/json" -d '{ "message": "my first req", "max_tokens": 30 }'
# StreamingResponse 接受一个异步生成器（async def 里带 yield）。
# text/event-stream 是 Server-Sent Events 格式


async def fetch_system_info():
    await asyncio.sleep(1)
    return {"gpu": "RTX 5060", "driver": "535"}

@app.get("/status")
async def get_status():
    # 并发获取多个信息
    import asyncio
    results = await asyncio.gather(
        fetch_system_info(),
        asyncio.sleep(0)  # 占位
    )
    return {"status": "ok", "info": results[0]}

# curl -N -X GET http://localhost:8000/status 
