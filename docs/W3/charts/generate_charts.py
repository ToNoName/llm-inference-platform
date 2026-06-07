import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Data from benchmark-w2-exploratory.md + llama-bench results
# ============================================================

labels = ['FP16', 'GPTQ-HF\n(desc_act=F)', 'GPTQ-Ours\n(desc_act=T)', 'AWQ-Ours\n(llm-compressor)', 'GGUF\nQ4_K_M', 'GGUF\nQ8_0']

# Model file sizes (GB)
model_size = [14.5, 5.3, 5.3, 5.2, 4.36, 7.54]

# vLLM VRAM (GB) - gpu_util=0.9 and 0.3
vram_09 = [21.67, 21.78, 21.45, 21.75, None, None]  # GGUF not tested on 4090D
vram_03 = [21.67, 7.33, 7.01, 7.30, None, None]

# vLLM TPOT (ms) - gpu_util=0.3 (more realistic)
tpot_gpu03 = [18.59, 18.95, 18.78, 20.45, None, None]

# vLLM throughput (tok/s) - gpu_util=0.3
tput_gpu03 = [53.98, 53.49, 53.71, 49.33, None, None]

# llama.cpp throughput (tok/s) - RTX 5060
tput_llamacpp = [None, None, None, None, 77.20, 45.18]

# ============================================================
# Figure 1: memory_by_quant.png
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Top: Model file size
colors_size = ['#2196F3', '#FF9800', '#FF9800', '#4CAF50', '#9C27B0', '#9C27B0']
bars1 = ax1.bar(labels, [s if s else 0 for s in model_size], color=colors_size, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars1, model_size):
    if val:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                 f'{val:.2f} GB', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_ylabel('Model File Size (GB)', fontsize=12)
ax1.set_title('Quantization: Model File Size Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 17)
ax1.axhline(y=8, color='red', linestyle='--', alpha=0.5, label='8GB GPU limit')
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Compression ratio annotations
comp_ratios = ['1.00x', '2.73x', '2.73x', '2.79x', '3.26x', '1.88x']
for bar, cr in zip(bars1, comp_ratios):
    ax1.text(bar.get_x() + bar.get_width()/2., 0.5, cr, ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

# Bottom: vLLM VRAM usage
x_pos = np.arange(4)
vllm_labels = ['FP16', 'GPTQ-HF\n(desc_act=F)', 'GPTQ-Ours\n(desc_act=T)', 'AWQ-Ours']

width = 0.35
bars_09 = ax2.bar(x_pos - width/2, [v if v else 0 for v in vram_09[:4]], width, label='gpu_util=0.9\n(default)', color='#2196F3', edgecolor='black', linewidth=0.5)
bars_03 = ax2.bar(x_pos + width/2, [v if v else 0 for v in vram_03[:4]], width, label='gpu_util=0.3\n(true VRAM)', color='#FF5722', edgecolor='black', linewidth=0.5)

for bar, val in zip(bars_09, vram_09[:4]):
    if val:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars_03, vram_03[:4]):
    if val:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)

ax2.set_ylabel('VRAM Usage (GB)', fontsize=12)
ax2.set_title('vLLM VRAM: gpu_util=0.9 vs 0.3 (RTX 4090D 24GB)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(vllm_labels, fontsize=10)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 25)
ax2.grid(axis='y', alpha=0.3)
ax2.text(0.02, 0.97, 'gpu_util=0.9: vLLM pre-allocates 90% VRAM\nfor KV Cache, masking model size diff',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('memory_by_quant.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: memory_by_quant.png')

# ============================================================
# Figure 2: latency_by_quant.png
# ============================================================
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 7))

# Left: vLLM TPOT (gpu_util=0.3)
vllm_tpots = tpot_gpu03[:4]
vllm_tp_labels = ['FP16', 'GPTQ-HF\n(desc_act=F)', 'GPTQ-Ours\n(desc_act=T)', 'AWQ-Ours']
colors_tpot = ['#2196F3', '#FF9800', '#FF9800', '#4CAF50']

bars3 = ax3.bar(vllm_tp_labels, [v if v else 0 for v in vllm_tpots], color=colors_tpot, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars3, vllm_tpots):
    if val:
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                 f'{val:.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_ylabel('Avg TPOT (ms)', fontsize=12)
ax3.set_title('vLLM Avg TPOT (gpu_util=0.3)\nRTX 4090D, 5 samples/group, single request', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 24)
ax3.axhline(y=18.59, color='gray', linestyle='--', alpha=0.5)
ax3.grid(axis='y', alpha=0.3)
ax3.text(0.02, 0.97, 'Single request: quantized models\nnot faster than FP16\n(memory-bound not reached)',
         transform=ax3.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Right: llama.cpp throughput
llm_labels = ['Q4_K_M\n(4.36 GB)', 'Q8_0\n(7.54 GB)']
llm_tput = [77.20, 45.18]
colors_llm = ['#9C27B0', '#9C27B0']

bars4 = ax4.bar(llm_labels, llm_tput, color=colors_llm, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars4, llm_tput):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
             f'{val:.1f} tok/s', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.set_ylabel('Token Generation Speed (tok/s)', fontsize=12)
ax4.set_title('llama.cpp Token Generation\nRTX 5060 8GB, -ngl 99', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 90)
ax4.grid(axis='y', alpha=0.3)
ax4.text(0.02, 0.97, 'Q4_K_M faster than Q8_0\non 8GB GPU: smaller model\n= more KV Cache space',
         transform=ax4.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.text(0.5, 0.01, 'DIFFERENT GPUs: vLLM=RTX4090D 24GB, llama.cpp=RTX5060 8GB. Speed NOT directly comparable.',
         ha='center', fontsize=11, color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('latency_by_quant.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: latency_by_quant.png')
print('Done! Run this script in WSL2 with: python generate_charts.py')