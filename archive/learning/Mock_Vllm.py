# Mock_Vllm.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio, json, time

app = FastAPI()

@app.post("/v1/chat/completions")
async def completions(request: dict):
    async def generate():
        for i in range(5):
            chunk = {
                "id": "cmpl-xxx",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "qwen",
                "choices": [{"delta": {"content": f" token_{i} "}, "index": 0}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.5)
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")