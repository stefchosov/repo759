# Code generated with assistance from Claude Code (Anthropic CLI)
# Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Usage: Final project — Task 1 throughput comparison plot
#
# Reads:
#   Final/Data/task1/scaling_task1.dat
#     columns: n  md5_serial_ms  md5_omp_ms  md5_gpu_ms
#              sha1_serial_ms  sha1_omp_ms  sha1_gpu_ms
#              sha256_serial_ms  sha256_omp_ms  sha256_gpu_ms
#
# Produces:
#   Final/Data/task1/task1_throughput.pdf
#   Final/Data/task1/task1_throughput.png

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DAT     = "Final/Data/task1/scaling_task1.dat"
OUT_PDF = "Final/Data/task1/task1_throughput.pdf"
OUT_PNG = "Final/Data/task1/task1_throughput.png"

if not os.path.exists(DAT):
    sys.exit(f"ERROR: {DAT} not found. Run task1_final_scaling.sh first.")

data = np.loadtxt(DAT, comments="#")
if data.ndim == 1:
    data = data.reshape(1, -1)

# Use the row with the largest n for the bar chart
row    = data[np.argmax(data[:, 0])]
n_best = row[0]

# columns: n  md5_s md5_o md5_g  sha1_s sha1_o sha1_g  s256_s s256_o s256_g
md5_s,  md5_o,  md5_g  = row[1], row[2], row[3]
sha1_s, sha1_o, sha1_g = row[4], row[5], row[6]
s256_s, s256_o, s256_g = row[7], row[8], row[9]

def throughput(n, ms):
    """hashes/sec → MH/s"""
    return (n / (ms / 1000.0)) / 1e6

thr = {
    "MD5":    {"Serial": throughput(n_best, md5_s),
               "OpenMP": throughput(n_best, md5_o),
               "GPU":    throughput(n_best, md5_g)},
    "SHA-1":  {"Serial": throughput(n_best, sha1_s),
               "OpenMP": throughput(n_best, sha1_o),
               "GPU":    throughput(n_best, sha1_g)},
    "SHA-256":{"Serial": throughput(n_best, s256_s),
               "OpenMP": throughput(n_best, s256_o),
               "GPU":    throughput(n_best, s256_g)},
}

has_gpu = True  # always present in combined dat

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
print(f"Saved  {OUT_PDF}")
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
