#!/usr/bin/env python3
"""W5 matplotlib charts for llama.cpp Q4_K_M benchmark analysis."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Font setup ─────────────────────────────────────────
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

FIG_STYLE = {"figsize": (10, 6), "dpi": 150}
PLOT_STYLE = {"linewidth": 2, "markersize": 8}

COLORS = {
    "128": "#2ecc71",
    "512": "#3498db",
    "1024": "#e74c3c",
    "vllm": "#9b59b6",
    "llama": "#e67e22",
    "p50": "#3498db",
    "p90": "#f39c12",
    "p99": "#e74c3c",
}


# ═══════════════════════════════════════════════════════
# Chart 1: TPOT P50 vs Concurrency (out=256, 3 lines)
# ═══════════════════════════════════════════════════════
def chart_tpot_vs_concurrency():
    c_levels = [1, 4, 8, 16]

    data = {
        "128":  [12.75, 31.64, 63.34, 119.33],
        "512":  [12.78, 37.38, 78.85, 147.61],
        "1024": [12.70, 41.24, 82.67, 153.11],
    }

    fig, ax = plt.subplots(**FIG_STYLE)
    for label, vals in data.items():
        ax.plot(c_levels, vals, marker="o", color=COLORS[label],
                label=f"{label} tokens", linewidth=PLOT_STYLE["linewidth"],
                markersize=PLOT_STYLE["markersize"])

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TPOT P50 (ms)")
    ax.set_title("TPOT P50 vs Concurrency (output=256)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(c_levels)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    path = os.path.join(OUTPUT_DIR, "tpot_vs_concurrency.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════
# Chart 2: TPOT P50 vs Input Tokens (c=1, out=256)
# ═══════════════════════════════════════════════════════
def chart_tpot_vs_input_tokens():
    input_levels = [128, 512, 1024]
    tpot_p50 = [12.75, 12.78, 12.70]
    tpot_p90 = [12.80, 12.82, 12.75]
    tpot_p99 = [13.70, 13.19, 13.25]

    fig, ax = plt.subplots(**FIG_STYLE)
    bar_width = 60
    x = np.array(input_levels)

    ax.bar(x - bar_width, tpot_p50, bar_width, color=COLORS["p50"], label="P50")
    ax.bar(x,           tpot_p90, bar_width, color=COLORS["p90"], label="P90")
    ax.bar(x + bar_width, tpot_p99, bar_width, color=COLORS["p99"], label="P99")

    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("TPOT vs Input Tokens (c=1, output=256)")
    ax.set_xticks(input_levels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(tpot_p99) * 1.3)

    path = os.path.join(OUTPUT_DIR, "tpot_vs_input_tokens.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════
# Chart 3: P50/P90/P99 spread (128 vs 1024, c=16, out=256)
# ═══════════════════════════════════════════════════════
def chart_p50_p90_p99_spread():
    labels = ["P50", "P90", "P99"]
    data_128  = [119.33, 124.64, 124.73]
    data_1024 = [153.11, 166.92, 167.00]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=FIG_STYLE["dpi"])

    for ax, data, title, color in [
        (ax1, data_128,  "128 tokens / 256 out", COLORS["128"]),
        (ax2, data_1024, "1024 tokens / 256 out", COLORS["1024"]),
    ]:
        bars = ax.bar(labels, data, color=[color, color, "#c0392b"], alpha=0.85, width=0.5)
        ax.set_title(title)
        ax.set_ylabel("TPOT (ms)")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # Annotate spread
    spread_128 = data_128[2] - data_128[0]
    spread_1024 = data_1024[2] - data_1024[0]
    fig.suptitle(
        f"P50/P90/P99 Spread (c=16)\n"
        f"128in gap={spread_128:.1f}ms ({spread_128/data_128[0]*100:.1f}%)  |  "
        f"1024in gap={spread_1024:.1f}ms ({spread_1024/data_1024[0]*100:.1f}%)",
        fontsize=13,
    )

    path = os.path.join(OUTPUT_DIR, "p50_p90_p99_spread_c16.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════
# Chart 4: llama.cpp vs vLLM concurrency scaling
# ═══════════════════════════════════════════════════════
def chart_llama_vs_vllm_scaling():
    # Normalized: TPOT / TPOT_c1
    c_vllm   = [1, 4, 8, 16, 30]
    c_llama  = [1, 4, 8, 16]

    tpot_vllm  = [6.18, 6.45, 6.71, 7.71, 8.69]
    tpot_llama = [12.75, 31.64, 63.34, 119.33]

    norm_vllm  = [v / tpot_vllm[0] for v in tpot_vllm]
    norm_llama = [v / tpot_llama[0] for v in tpot_llama]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=FIG_STYLE["dpi"])

    # Absolute
    ax1.plot(c_vllm, tpot_vllm, marker="o", color=COLORS["vllm"],
             label="vLLM GPTQ (4090D)", linewidth=PLOT_STYLE["linewidth"], markersize=PLOT_STYLE["markersize"])
    ax1.plot(c_llama, tpot_llama, marker="s", color=COLORS["llama"],
             label="llama.cpp Q4_K_M (5060)", linewidth=PLOT_STYLE["linewidth"], markersize=PLOT_STYLE["markersize"])
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("TPOT P50 (ms)")
    ax1.set_title("Absolute TPOT")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Normalized
    ax2.plot(c_vllm, norm_vllm, marker="o", color=COLORS["vllm"],
             label="vLLM GPTQ (4090D)", linewidth=PLOT_STYLE["linewidth"], markersize=PLOT_STYLE["markersize"])
    ax2.plot(c_llama, norm_llama, marker="s", color=COLORS["llama"],
             label="llama.cpp Q4_K_M (5060)", linewidth=PLOT_STYLE["linewidth"], markersize=PLOT_STYLE["markersize"])
    ax2.set_xlabel("Concurrency")
    ax2.set_ylabel("TPOT (normalized to c=1)")
    ax2.set_title("Normalized TPOT (c=1 = 1.0)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle(
        "llama.cpp vs vLLM Concurrency Scaling (directional, different GPUs)",
        fontsize=13, fontweight="bold",
    )

    path = os.path.join(OUTPUT_DIR, "llama_vs_vllm_scaling.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    chart_tpot_vs_concurrency()
    chart_tpot_vs_input_tokens()
    chart_p50_p90_p99_spread()
    chart_llama_vs_vllm_scaling()
    print(f"\nAll charts saved to {OUTPUT_DIR}/")
