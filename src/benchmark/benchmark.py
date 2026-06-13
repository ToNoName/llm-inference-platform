#!/usr/bin/env python3
"""
推理后端压测脚本

用法：
  # 直接压llama.cpp后端
  python benchmark.py --backend llama --url http://localhost:8002/v1/chat/completions \
      --concurrency 1,4,8 --num-requests 30

  # 走网关压测
  python benchmark.py --backend gateway --url http://localhost:8000/v1/chat/completions \
      --concurrency 1,4,8 --num-requests 30

  # 直接压vLLM后端（AutoDL）
  python benchmark.py --backend vllm --url http://localhost:8001/v1/chat/completions \
      --concurrency 1,4,8,16,30 --num-requests 30
"""

import argparse
import asyncio
import csv
import json
import statistics
import time
from datetime import datetime, timezone

import httpx

# ---------- 预定义prompt模板 ----------

PROMPTS = {
    64: "请简单介绍一下人工智能的发展历史。",
    128: "请详细解释深度学习中的注意力机制（Attention Mechanism），包括它的原理、作用以及在Transformer模型中的应用。请举例说明。",
    512: (
        "请写一篇关于大语言模型推理部署优化的技术文章，包括以下内容："
        "1. 推理延迟优化的核心策略（KV Cache管理、批处理、量化）；"
        "2. 显存管理的关键技术（PagedAttention、显存预分配策略）；"
        "3. 不同部署引擎的对比（vLLM、llama.cpp、SGLang）；"
        "4. 生产环境中的监控和运维实践。每个部分请给出具体的技术细节和性能数据。"
    ),
    1024: (
        "请写一篇全面的技术白皮书，主题为《从零到生产：大模型推理系统设计与实践》。内容需要涵盖："
        "一、背景与挑战（模型规模增长趋势、推理延迟与吞吐的trade-off、显存带宽瓶颈分析）；"
        "二、核心优化技术（量化方法GPTQ/AWQ/GGUF的原理与适用场景、"
        "KV Cache管理的演进从静态分配到PagedAttention、Continuous Batching的原理与实现、"
        "推测解码Speculative Decoding）；"
        "三、部署架构设计（推理引擎选型vLLM vs llama.cpp vs SGLang、"
        "多卡并行策略TP/PP、API网关设计与负载均衡、容器化部署最佳实践）；"
        "四、性能调优与监控（关键性能指标TTFT/TPOT/QPS的定义与测量、"
        "gpu_memory_utilization参数调优、Prometheus+Grafana监控方案）；"
        "五、案例研究（7B模型在RTX 4090上的部署实践、量化前后性能对比数据、并发压测结果分析）。"
        "请确保每个部分都有具体数据和技术细节。"
    ),
}

DEFAULT_PROMPT = PROMPTS[128]


def prepare_payload(
    prompt: str, max_tokens: int = 256, model: str = "qwen", stream: bool = False
) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": stream,
    }


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    timeout: float = 120.0,
) -> dict:
    start = time.monotonic()
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        e2e_latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code != 200:
            return {
                "status": "error",
                "status_code": resp.status_code,
                "e2e_latency_ms": e2e_latency_ms,
            }

        data = resp.json()
        usage = data.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        backend_latency_ms = e2e_latency_ms

        ttft_ms = backend_latency_ms
        tpot_ms = (backend_latency_ms / output_tokens) if output_tokens > 0 else 0

        return {
            "status": "success",
            "status_code": 200,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_latency_ms": e2e_latency_ms,
            "backend_latency_ms": backend_latency_ms,
            "tokens_per_second": (output_tokens / (e2e_latency_ms / 1000)) if e2e_latency_ms > 0 else 0,
        }
    except httpx.TimeoutException:
        return {"status": "timeout", "e2e_latency_ms": (time.monotonic() - start) * 1000}
    except httpx.ConnectError:
        return {"status": "connection_error", "e2e_latency_ms": (time.monotonic() - start) * 1000}
    except Exception as e:
        return {"status": "error", "error": str(e), "e2e_latency_ms": (time.monotonic() - start) * 1000}


async def run_benchmark(
    url: str,
    backend: str,
    model: str,
    quant: str,
    concurrency: int,
    num_requests: int,
    prompt_tokens_target: int,
    max_tokens: int,
    warmup: int = 1,
    timeout: float = 120.0,
) -> list:
    prompt = PROMPTS.get(prompt_tokens_target, DEFAULT_PROMPT)
    payload = prepare_payload(prompt, max_tokens=max_tokens, model=model)
    limits = httpx.Limits(max_connections=concurrency + 10)

    # 预热
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for _ in range(warmup):
            await send_request(client, url, payload, timeout)

    # 正式压测
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request():
        async with semaphore:
            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                return await send_request(client, url, payload, timeout)

    tasks = [bounded_request() for _ in range(num_requests)]
    raw_results = await asyncio.gather(*tasks)
    results = list(raw_results)
    return results


def percentile(data, p):
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def compute_stats(results: list) -> dict:
    success = [r for r in results if r.get("status") == "success"]
    if not success:
        return {"num_success": 0, "num_total": len(results)}

    output_tokens_list = [r["output_tokens"] for r in success]
    ttft_list = [r["ttft_ms"] for r in success]
    tpot_list = [r["tpot_ms"] for r in success]
    e2e_list = [r["e2e_latency_ms"] for r in success]
    tps_list = [r["tokens_per_second"] for r in success]

    return {
        "num_success": len(success),
        "num_total": len(results),
        "num_errors": len(results) - len(success),
        "output_tokens_mean": round(statistics.mean(output_tokens_list), 1),
        "ttft_ms_p50": round(percentile(ttft_list, 50), 2),
        "ttft_ms_p90": round(percentile(ttft_list, 90), 2),
        "ttft_ms_p99": round(percentile(ttft_list, 99), 2),
        "tpot_ms_p50": round(percentile(tpot_list, 50), 2),
        "tpot_ms_p90": round(percentile(tpot_list, 90), 2),
        "tpot_ms_p99": round(percentile(tpot_list, 99), 2),
        "e2e_ms_p50": round(percentile(e2e_list, 50), 2),
        "e2e_ms_p90": round(percentile(e2e_list, 90), 2),
        "e2e_ms_p99": round(percentile(e2e_list, 99), 2),
        "tps_mean": round(statistics.mean(tps_list), 2),
        "tps_p50": round(percentile(tps_list, 50), 2),
    }


CSV_FIELDNAMES = [
    "run_id",
    "timestamp",
    "backend",
    "model",
    "quant",
    "input_tokens",
    "output_tokens",
    "concurrency",
    "ttft_ms",
    "tpot_ms",
    "e2e_latency_ms",
    "backend_latency_ms",
    "tokens_per_second",
    "status",
]


def write_csv(results: list, filepath: str):
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            row = {
                "run_id": r.get("run_id", ""),
                "timestamp": r.get("timestamp", ""),
                "backend": r.get("backend", ""),
                "model": r.get("model", ""),
                "quant": r.get("quant", ""),
                "input_tokens": r.get("input_tokens", r.get("prompt_tokens", "")),
                "output_tokens": r.get("output_tokens", ""),
                "concurrency": r.get("concurrency", ""),
                "ttft_ms": r.get("ttft_ms", ""),
                "tpot_ms": r.get("tpot_ms", ""),
                "e2e_latency_ms": r.get("e2e_latency_ms", ""),
                "backend_latency_ms": r.get("backend_latency_ms", ""),
                "tokens_per_second": r.get("tokens_per_second", ""),
                "status": r.get("status", ""),
            }
            writer.writerow(row)


async def main():
    parser = argparse.ArgumentParser(description="LLM推理后端压测脚本")
    parser.add_argument("--backend", required=True, choices=["vllm", "llama", "gateway"])
    parser.add_argument("--url", required=True, help="后端URL")
    parser.add_argument("--model", default="qwen", help="模型名")
    parser.add_argument("--quant", default="unknown", help="量化方式")
    parser.add_argument("--concurrency", default="1", help="并发数，逗号分隔")
    parser.add_argument("--num-requests", type=int, default=30, help="每组请求数")
    parser.add_argument("--input-tokens", default="128", help="输入token数目标，逗号分隔")
    parser.add_argument("--max-tokens", type=int, default=256, help="输出最大token数")
    parser.add_argument("--warmup", type=int, default=1, help="预热次数")
    parser.add_argument("--timeout", type=float, default=120.0, help="单请求超时(秒)")
    parser.add_argument("--output", default="benchmark_results.csv", help="CSV输出路径")

    args = parser.parse_args()

    concurrency_list = [int(x) for x in args.concurrency.split(",")]
    input_tokens_list = [int(x) for x in args.input_tokens.split(",")]

    all_results = []
    run_id_prefix = datetime.now(timezone.utc).strftime("%Y%m%d") + f"_{args.backend}_{args.quant}"

    for input_tok in input_tokens_list:
        for conc in concurrency_list:
            run_id = f"{run_id_prefix}_{input_tok}in_{args.max_tokens}out_c{conc}"
            timestamp = datetime.now(timezone.utc).isoformat()
            print(f"\n{'=' * 60}")
            print(
                f"Running: {run_id} | concurrency={conc} | "
                f"input_tokens~={input_tok} | max_tokens={args.max_tokens}"
            )
            print(f"{'=' * 60}")

            results = await run_benchmark(
                url=args.url,
                backend=args.backend,
                model=args.model,
                quant=args.quant,
                concurrency=conc,
                num_requests=args.num_requests,
                prompt_tokens_target=input_tok,
                max_tokens=args.max_tokens,
                warmup=args.warmup,
                timeout=args.timeout,
            )

            for r in results:
                r["run_id"] = run_id
                r["timestamp"] = timestamp
                r["backend"] = args.backend
                r["model"] = args.model
                r["quant"] = args.quant
                r["concurrency"] = conc
                r["input_tokens"] = input_tok

            all_results.extend(results)

            stats = compute_stats(results)
            print(f"\n--- Results for {run_id} ---")
            print(f"  Success: {stats.get('num_success', 0)}/{stats.get('num_total', 0)}")
            if stats.get("num_success", 0) > 0:
                print(f"  TTFT: P50={stats['ttft_ms_p50']}ms P90={stats['ttft_ms_p90']}ms P99={stats['ttft_ms_p99']}ms")
                print(f"  TPOT: P50={stats['tpot_ms_p50']}ms P90={stats['tpot_ms_p90']}ms P99={stats['tpot_ms_p99']}ms")
                print(f"  E2E:  P50={stats['e2e_ms_p50']}ms P90={stats['e2e_ms_p90']}ms")
                print(f"  TPS:  mean={stats['tps_mean']} P50={stats['tps_p50']}")

    write_csv(all_results, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())