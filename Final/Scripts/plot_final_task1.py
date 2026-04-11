# Code generated with assistance from Claude Code (Anthropic CLI)
# Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Usage: Final project — Task 1 throughput comparison plot
#
# Reads:
#   Final/Data/scaling_task1_cpu.dat  (serial + OpenMP)
#   Final/Data/scaling_task1_gpu.dat  (GPU — optional, skipped if absent)
#
# Produces:
#   Final/Data/task1_throughput.pdf
#   Final/Data/task1_throughput.png
#
# Graph: grouped bar chart (log-scale y-axis)
#   X: algorithm (MD5, SHA-1, SHA-256)
#   Groups: Serial / OpenMP / GPU
#   Y: throughput in MH/s  (log scale — covers 3 orders of magnitude)
#   Evaluated at the largest n in the dat file.

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CPU_DAT  = "Final/Data/scaling_task1_cpu.dat"
GPU_DAT  = "Final/Data/scaling_task1_gpu.dat"
OUT_PDF  = "Final/Data/task1_throughput.pdf"
OUT_PNG  = "Final/Data/task1_throughput.png"

# ── load CPU data ─────────────────────────────────────────────────────────────

if not os.path.exists(CPU_DAT):
    sys.exit(f"ERROR: {CPU_DAT} not found. Run task1_final_scaling.sh first.")

cpu_data = np.loadtxt(CPU_DAT, comments="#")
if cpu_data.ndim == 1:
    cpu_data = cpu_data.reshape(1, -1)

# Use the row with the largest n for the bar chart
row = cpu_data[np.argmax(cpu_data[:, 0])]
n_best = row[0]

# columns: n  md5_serial  md5_omp  sha1_serial  sha1_omp  sha256_serial  sha256_omp
md5_serial_ms,  md5_omp_ms  = row[1], row[2]
sha1_serial_ms, sha1_omp_ms = row[3], row[4]
s256_serial_ms, s256_omp_ms = row[5], row[6]

def throughput(n, ms):
    """hashes per second → MH/s"""
    return (n / (ms / 1000.0)) / 1e6

thr = {
    "MD5":    {"Serial": throughput(n_best, md5_serial_ms),
               "OpenMP": throughput(n_best, md5_omp_ms)},
    "SHA-1":  {"Serial": throughput(n_best, sha1_serial_ms),
               "OpenMP": throughput(n_best, sha1_omp_ms)},
    "SHA-256":{"Serial": throughput(n_best, s256_serial_ms),
               "OpenMP": throughput(n_best, s256_omp_ms)},
}

# ── load GPU data (optional) ──────────────────────────────────────────────────

has_gpu = os.path.exists(GPU_DAT)
if has_gpu:
    gpu_data = np.loadtxt(GPU_DAT, comments="#")
    if gpu_data.ndim == 1:
        gpu_data = gpu_data.reshape(1, -1)
    grow = gpu_data[np.argmax(gpu_data[:, 0])]
    # columns: n  md5_gpu_ms  sha1_gpu_ms  sha256_gpu_ms
    gn = grow[0]
    thr["MD5"]["GPU"]     = throughput(gn, grow[1])
    thr["SHA-1"]["GPU"]   = throughput(gn, grow[2])
    thr["SHA-256"]["GPU"] = throughput(gn, grow[3])

# ── plot ──────────────────────────────────────────────────────────────────────

algos  = ["MD5", "SHA-1", "SHA-256"]
modes  = ["Serial", "OpenMP", "GPU"] if has_gpu else ["Serial", "OpenMP"]
colors = {"Serial": "#4c72b0", "OpenMP": "#dd8452", "GPU": "#55a868"}

n_algos = len(algos)
n_modes = len(modes)
bar_w   = 0.22
offsets = np.arange(n_modes) - (n_modes - 1) / 2.0

fig, ax = plt.subplots(figsize=(9, 5.5))

for mi, mode in enumerate(modes):
    vals  = [thr[a].get(mode, 0.0) for a in algos]
    x_pos = np.arange(n_algos) + offsets[mi] * bar_w
    bars  = ax.bar(x_pos, vals, width=bar_w, label=mode,
                   color=colors[mode], edgecolor="white", linewidth=0.5)

    # Annotate bars with value (e.g. "1.2 GH/s")
    for bar, v in zip(bars, vals):
        if v <= 0:
            continue
        label = f"{v/1000:.1f} GH/s" if v >= 1000 else f"{v:.0f} MH/s"
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.15,
                label, ha="center", va="bottom", fontsize=7, rotation=45)

ax.set_yscale("log")
ax.set_xticks(np.arange(n_algos))
ax.set_xticklabels(algos, fontsize=12)
ax.set_ylabel("Throughput (MH/s, log scale)", fontsize=11)
ax.set_xlabel("Hash Algorithm", fontsize=11)
ax.set_title(
    f"CPU Throughput — Serial vs OpenMP{' vs GPU' if has_gpu else ''}\n"
    f"(n = {int(n_best):,} hashes, log y-axis)",
    fontsize=12
)
ax.legend(fontsize=10)
ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT_PDF, dpi=150)
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved {OUT_PDF}")
print(f"Saved {OUT_PNG}")

# ── print speedup table ───────────────────────────────────────────────────────

print(f"\nThroughput summary (n = {int(n_best):,})")
print(f"{'':12s}  {'Serial':>12s}  {'OpenMP':>12s}", end="")
if has_gpu:
    print(f"  {'GPU':>12s}  {'GPU/Serial':>12s}", end="")
print()
for a in algos:
    serial = thr[a]["Serial"]
    omp    = thr[a]["OpenMP"]
    line   = f"{a:12s}  {serial:>10.1f} MH/s  {omp:>10.1f} MH/s"
    if has_gpu:
        gpu = thr[a]["GPU"]
        line += f"  {gpu:>10.1f} MH/s  {gpu/serial:>10.1f}x"
    print(line)
