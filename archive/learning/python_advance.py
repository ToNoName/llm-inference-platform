
######## ----------------  1.类型注解示例 ----------------- ########
# #基础变量类型注解
# name: str = "推理部署工程师"
# gpu_num: int = 1
# util_rate: float = 0.75

# #函数参数+返回值注解
# def calc_infer_delay(input_len: int, output_len: int)    -> float:
#     """计算推理延迟"""
#     return (input_len + output_len) * 0.015

# #复杂类型注解
# from typing import List,Dict
# model_list: List[str] = ["qwen2.5-3b", "qwen2.5-7b"]
# infer_config: Dict[str,int] = {"ctx_len": 4096, "batch_size": 16}

# #显存计算函数
# def calc_gpu_memory(model_params: float, quant_size: int ,ctx_size: int) -> float:
#     """计算显存占用"""
#     #模型权重显存
#     model_memory = model_params * quant_size / 8
#     #KV缓存显存
#     kv_cache_memory = ctx_size * 0.0001
#     #总显存
#     total_memory_GB = (model_memory + kv_cache_memory) / 1024 ** 3  # 转换为GB


#     return total_memory_GB  # 转换为GB

# if __name__ == "__main__":
#     mem = calc_gpu_memory(3e9, 4, 4096)
#     print(f"显存占用: {mem:.2f} GB")


#############----------------  2.dataclass数据类示例 ----------------- #############
# from dataclasses import dataclass

# @dataclass
# class ModelConfig:
#     """模型配置数据类"""
#     model_name: str
#     num_params: float  # 单位：亿
#     quantization_bits: int
#     context_length: int

#     def calc_gpu_memory(self) -> float:
#         """计算显存占用"""
#         # 模型权重显存
#         model_memory = self.num_params * 1e8 * self.quantization_bits / 8  # 转换为字节
#         # KV缓存显存
#         kv_cache_memory = self.context_length * 0.0001 * 1024 ** 3  # 转换为字节
#         # 总显存
#         total_memory_GB = (model_memory + kv_cache_memory) / 1024 ** 3  # 转换为GB
#         return total_memory_GB

# config = ModelConfig(model_name="qwen2.5-3b", num_params=3, quantization_bits=4, context_length=4096)
# print(f"模型配置: {config}") # f = f-string = 格式化字符串，自动生成的 __repr__ 方法会输出所有字段的值
# print(f"显存占用: {config.calc_gpu_memory():.2f} GB")


# #############----------------  3.装饰器示例 ----------------- #############

# def log_call(func):  #接收函数 
#     """一个简单的装饰器，用于记录函数调用"""
#     def wrapper(*args, **kwargs): #接收任意参数
#         print(f"调用函数: {func.__name__}，参数: {args}, {kwargs}")
#         result = func(*args, **kwargs) #调用原函数并获取结果
#         print(f"函数返回: {result}")
#         return result
#     return wrapper #返回包装后的函数

# @log_call      # 使用装饰器 等价于: add = log_call(add)
# def add(a: int, b: int) -> int:
#     """一个简单的加法函数"""
#     return a + b

# add(3, 5)


# #单层写法
# def log_call2(func):
#     print("绑定单层装饰器成功")
#     return func   # 只返回函数，不调用

# @log_call2
# def add2(a: int, b: int) -> int:
#     """一个简单的加法函数"""
#     return a + b

# add2(4, 6)


# #############----------------  4. async / await示例 ----------------- #############

# import asyncio
# async def async_add(a: int, b: int) -> int: #async def = 定义异步函数（协程）不能直接调用，必须交给 asyncio 运行
#     """一个异步加法函数，模拟耗时操作"""
#     print(f"开始计算: {a} + {b}")
#     await asyncio.sleep(1)  # 模拟耗时操作,异步等待 1 秒（不阻塞 CPU）, 让出控制权给事件循环，允许其他任务运行
#     result = a + b
#     print(f"计算完成: {a} + {b} = {result}")
#     return result

# async def main():
#     # 创建多个异步任务
#     tasks = [async_add(i, i+1) for i in range(5)]
#     # 等待所有任务完成并获取结果
#     results = await asyncio.gather(*tasks)  # asyncio.gather = 并行等待多个协程完成，返回结果列表, gather不认识列表,所以 *tasks = 解包任务列表，传递给 gather
#     print(f"所有计算结果: {results}")

# if __name__ == "__main__":
#     asyncio.run(main())


#############----------------  5. 完整示例 ----------------- #############
import asyncio
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from functools import wraps


#----- dataaclass 请求/响应结构体
@dataclass
class ChatRequest:
    """聊天请求数据类"""
    prompt: str
    max_tokens: int = 256
    temperature: Optional[float] = None

@dataclass
class ChatResponse:
    text: str
    tokens_generated: int
    latency_ms: float

#装饰器 计时+日志
def measure_time(func):
    """一个装饰器，用于测量函数执行时间"""
    @wraps(func) # 保持原函数的元数据（如函数名、文档字符串等）
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        Latency = (end_time - start_time) * 1000  # 转换为毫秒
        print(f"函数 {func.__name__} 执行时间: {Latency:.2f} ms")
        return result
    return wrapper

#模拟推理函数 

async def fake_llm_generate(prompt: str, max_tokens: int, temperature: Optional[float]) -> str:
    """一个模拟的LLM生成函数，异步执行"""
    tokens = ["这是", "一个", "模拟的", "LLM", "生成的", "文本。"] * (max_tokens // 6)  # 模拟生成的tokens数量
    generated = []
    for i,token in enumerate(tokens):
        await asyncio.sleep(0.01)  # 模拟每个token生成的时间
        generated.append(token)
    return "".join(generated)

# 带装饰器的推理服务
@measure_time
async def chat_service(request: ChatRequest) -> ChatResponse:
    """一个带装饰器的聊天服务函数"""
    generated_text = await fake_llm_generate(request.prompt, request.max_tokens, request.temperature)
    response = ChatResponse(
        text=generated_text,
        tokens_generated=len(generated_text.split()),
        latency_ms=0.0  # 这里暂时不计算实际延迟，装饰器会输出
    )
    return response

# 开发请求测试
async def batch_inference(requests: List[ChatRequest]) -> List[ChatResponse]:
    """批量推理函数，处理多个请求"""
    tasks = [chat_service(req) for req in requests]  # 创建任务列表
    responses = await asyncio.gather(*tasks)  # 并行等待所有任务完成
    return responses

async def main():
    # 创建多个请求
    requests = [
        ChatRequest(prompt="你好，世界！", max_tokens=12),
        ChatRequest(prompt="请介绍一下你自己。", max_tokens=18, temperature=0.7),
        ChatRequest(prompt="今天天气怎么样？", max_tokens=10)
    ]
    # 批量推理
    responses = await batch_inference(requests)
    for i, response in enumerate(responses):
        print(f"响应 {i+1}: {response}")

if __name__ == "__main__":
    asyncio.run(main())



