# Code generated with assistance from Claude Code (Anthropic CLI)
# Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Usage: Final project — Task 2 tpb scaling plot
#
# Reads:  Final/Data/task2/scaling_task2.dat
#   columns: tpb  md5_gpu_ms  sha1_gpu_ms  sha256_gpu_ms
#
# Produces:
#   Final/Data/task2/task2_tpb.pdf
#   Final/Data/task2/task2_tpb.png
#
# Graph: line chart — tpb (x, log2 scale) vs throughput MH/s (y)
#        one line per algorithm

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DAT     = "Final/Data/task2/scaling_task2.dat"
OUT_PDF = "Final/Data/task2/task2_tpb.pdf"
OUT_PNG = "Final/Data/task2/task2_tpb.png"

if not os.path.exists(DAT):
    sys.exit(f"ERROR: {DAT} not found. Run task2_final_scaling.sh first.")

data = np.loadtxt(DAT, comments="#")
if data.ndim == 1:
    data = data.reshape(1, -1)

N = 10_000_000

tpb      = data[:, 0].astype(int)
md5_thr  = (N / (data[:, 1] / 1000)) / 1e6
sha1_thr = (N / (data[:, 2] / 1000)) / 1e6
s256_thr = (N / (data[:, 3] / 1000)) / 1e6

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(tpb, md5_thr,  "o-", color="#4c72b0", label="MD5",     linewidth=2, markersize=7)
ax.plot(tpb, sha1_thr, "s-", color="#dd8452", label="SHA-1",   linewidth=2, markersize=7)
ax.plot(tpb, s256_thr, "^-", color="#55a868", label="SHA-256", linewidth=2, markersize=7)

ax.set_xscale("log", base=2)
ax.set_xticks(tpb)
ax.set_xticklabels([str(t) for t in tpb])
ax.set_xlabel("Threads per Block", fontsize=12)
ax.set_ylabel("Throughput (MH/s)", fontsize=12)
ax.set_title(f"GPU Throughput vs. Threads per Block\n(n = {N:,} hashes)", fontsize=13)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT_PDF, dpi=150)
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved {OUT_PDF}")
print(f"Saved {OUT_PNG}")

# Print summary table
print(f"\n{'tpb':>6}  {'MD5 MH/s':>10}  {'SHA-1 MH/s':>12}  {'SHA-256 MH/s':>14}")
for i in range(len(tpb)):
    print(f"{tpb[i]:>6}  {md5_thr[i]:>10.1f}  {sha1_thr[i]:>12.1f}  {s256_thr[i]:>14.1f}")
