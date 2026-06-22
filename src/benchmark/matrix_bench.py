#!/usr/bin/env python3
"""Automated benchmark matrix runner.
Runs all combinations of input_tokens x output_tokens x concurrency
against a single backend. Supports --resume to skip completed groups.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

from benchmark.benchmark import run_benchmark, compute_stats, CSV_FIELDNAMES

# ── Matrix ──────────────────────────────────────────────
INPUT_TOKENS = [128, 512, 1024]
OUTPUT_TOKENS = [64, 256]
CONCURRENCY = [1, 4, 8, 16]
NUM_REQUESTS = 30
WARMUP = 1
TIMEOUT = 180

# ── Backend ─────────────────────────────────────────────
BACKEND = "llama"
URL = "http://localhost:8002/v1/chat/completions"
MODEL = "qwen2.5-7b-instruct-Q4_K_M"
QUANT = "gguf-q4_k_m"

# ── Output ──────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "W5")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline-llama-matrix.csv")


def total_groups():
    return len(INPUT_TOKENS) * len(OUTPUT_TOKENS) * len(CONCURRENCY)


def completed_groups(filepath):
    """Parse CSV for completed run_ids (for resume support)."""
    if not os.path.exists(filepath):
        return set()
    done = set()
    with open(filepath, encoding="utf-8-sig") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            vals = line.strip().split(",")
            header_fields = header.strip().split(",")
            if "run_id" in header_fields:
                idx = header_fields.index("run_id")
                if idx < len(vals):
                    done.add(vals[idx])
    return done


def append_csv_header(filepath):
    """Write CSV header if file does not exist."""
    if os.path.exists(filepath):
        return
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        f.write(",".join(CSV_FIELDNAMES) + "\n")


def append_csv_rows(filepath, rows):
    """Append result rows to existing CSV."""
    with open(filepath, "a", newline="", encoding="utf-8-sig") as f:
        for r in rows:
            vals = [
                str(r.get("run_id", "")),
                str(r.get("timestamp", "")),
                str(r.get("backend", "")),
                str(r.get("model", "")),
                str(r.get("quant", "")),
                str(r.get("input_tokens", r.get("prompt_tokens", ""))),
                str(r.get("output_tokens", "")),
                str(r.get("concurrency", "")),
                str(r.get("ttft_ms", "")),
                str(r.get("tpot_ms", "")),
                str(r.get("e2e_latency_ms", "")),
                str(r.get("backend_latency_ms", "")),
                str(r.get("tokens_per_second", "")),
                str(r.get("status", "")),
            ]
            f.write(",".join(vals) + "\n")


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resume = "--resume" in sys.argv
    skip_ids = completed_groups(OUTPUT_FILE) if resume else set()

    # ── Warmup ──────────────────────────────────────────
    print("Warming up (128in/64out, c=1)...", end=" ", flush=True)
    warm_result = await run_benchmark(
        url=URL, backend=BACKEND, model=MODEL, quant=QUANT,
        concurrency=1, num_requests=1,
        prompt_tokens_target=128, max_tokens=64,
        warmup=0, timeout=TIMEOUT,
    )
    ok = sum(1 for r in warm_result if r.get("status") == "success")
    print(f"{ok}/1 success")
    if ok == 0:
        sys.exit("ERROR: Warmup failed. Ensure backend is running on " + URL)

    # ── Matrix loop ─────────────────────────────────────
    total = total_groups()
    idx = 0
    skipped = 0
    errors = 0

    for input_tok in INPUT_TOKENS:
        for output_tok in OUTPUT_TOKENS:
            for conc in CONCURRENCY:
                idx += 1
                run_id = (
                    f"matrix_{BACKEND}_{QUANT}_{input_tok}in_{output_tok}out_c{conc}"
                )

                if resume and run_id in skip_ids:
                    print(f"[{idx:>2}/{total}] SKIP {run_id}")
                    skipped += 1
                    continue

                print(
                    f"[{idx:>2}/{total}] {input_tok:>4}in {output_tok:>3}out c={conc:<2}",
                    end="  ", flush=True,
                )

                try:
                    results = await run_benchmark(
                        url=URL, backend=BACKEND, model=MODEL, quant=QUANT,
                        concurrency=conc, num_requests=NUM_REQUESTS,
                        prompt_tokens_target=input_tok, max_tokens=output_tok,
                        warmup=0, timeout=TIMEOUT,
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    errors += 1
                    continue

                # Attach metadata
                ts = datetime.now(timezone.utc).isoformat()
                for r in results:
                    r["run_id"] = run_id
                    r["timestamp"] = ts
                    r["backend"] = BACKEND
                    r["model"] = MODEL
                    r["quant"] = QUANT
                    r["concurrency"] = conc
                    r["input_tokens"] = input_tok

                stats = compute_stats(results)
                n_ok = stats.get("num_success", 0)

                if n_ok > 0:
                    print(
                        f"OK n={n_ok:<2}  "
                        f"TPOT_P50={stats['tpot_ms_p50']:>6.2f}ms  "
                        f"TPOT_P90={stats['tpot_ms_p90']:>6.2f}ms  "
                        f"E2E_P50={stats['e2e_ms_p50']:>7.0f}ms"
                    )
                else:
                    print(f"FAIL 0/{NUM_REQUESTS}")
                    errors += 1

                # Incremental write
                append_csv_header(OUTPUT_FILE)
                append_csv_rows(OUTPUT_FILE, results)

    # ── Summary ─────────────────────────────────────────
    ran = idx - skipped
    print(f"\n=== Done: {ran} groups, {errors} errors, {skipped} skipped ===")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
