# Code generated with assistance from Claude Code (Anthropic CLI)
# Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Usage: Final project — Task 3 collision search plot
#
# Reads:  Final/Data/task3/scaling_task3.dat
#   columns: algo  bits  cpu_count  cpu_ms  gpu_batch  gpu_ms  expected
#
# Produces:
#   Final/Data/task3/task3_time.pdf / .png
#     — Time (ms) vs truncated bits, log scale, CPU vs GPU per algorithm
#   Final/Data/task3/task3_birthday.pdf / .png
#     — Actual count / expected count vs bits (birthday paradox validation)

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DAT = "Final/Data/task3/scaling_task3.dat"
if not os.path.exists(DAT):
    sys.exit(f"ERROR: {DAT} not found. Run task3_final_scaling.sh first.")

# Parse .dat file (skip comment lines)
rows = []
with open(DAT) as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        algo, bits = parts[0], int(parts[1])
        cpu_count, cpu_ms = int(parts[2]), float(parts[3])
        gpu_batch, gpu_ms = int(parts[4]), float(parts[5])
        expected           = int(parts[6])
        rows.append((algo, bits, cpu_count, cpu_ms, gpu_batch, gpu_ms, expected))

algos    = ["md5", "sha1", "sha256"]
labels   = {"md5": "MD5", "sha1": "SHA-1", "sha256": "SHA-256"}
colors   = {"md5": "#4c72b0", "sha1": "#dd8452", "sha256": "#55a868"}
bits_all = sorted(set(r[1] for r in rows))

def get(algo, field):
    idx = {"cpu_count":2, "cpu_ms":3, "gpu_batch":4, "gpu_ms":5, "expected":6}[field]
    return np.array([r[idx] for r in rows if r[0] == algo])

# ── Plot 1: Time vs bits ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for algo in algos:
    c = colors[algo]
    lbl = labels[algo]
    cpu_ms = get(algo, "cpu_ms")
    gpu_ms = get(algo, "gpu_ms")
    ax.plot(bits_all, cpu_ms, "o-",  color=c, linewidth=2, markersize=7,
            label=f"{lbl} CPU")
    ax.plot(bits_all, gpu_ms, "s--", color=c, linewidth=2, markersize=7, alpha=0.75,
            label=f"{lbl} GPU")

ax.set_yscale("log")
ax.set_xlabel("Truncated Hash Bits", fontsize=12)
ax.set_ylabel("Time to Find Collision (ms)", fontsize=12)
ax.set_title("Truncated Collision Search: CPU vs GPU\n(solid=CPU, dashed=GPU)", fontsize=13)
ax.set_xticks(bits_all)
ax.legend(fontsize=9, ncol=2)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Final/Data/task3/task3_time.pdf", dpi=150)
fig.savefig("Final/Data/task3/task3_time.png", dpi=150)
print("Saved Final/Data/task3/task3_time.pdf")
plt.close(fig)

# ── Plot 2: Birthday paradox validation ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(bits_all))
width = 0.25
for di, algo in enumerate(algos):
    cpu_count = get(algo, "cpu_count")
    expected  = get(algo, "expected")
    ratio = cpu_count / expected
    ax.bar(x + (di - 1) * width, ratio, width,
           label=labels[algo], color=colors[algo], alpha=0.85)

ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Expected (theory)")
ax.set_xlabel("Truncated Hash Bits", fontsize=12)
ax.set_ylabel("Actual Count / Expected Count", fontsize=12)
ax.set_title("Birthday Paradox Validation\n(actual CPU collision count vs. theoretical √(π/2)·2^(bits/2))",
             fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in bits_all])
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Final/Data/task3/task3_birthday.pdf", dpi=150)
fig.savefig("Final/Data/task3/task3_birthday.png", dpi=150)
print("Saved Final/Data/task3/task3_birthday.pdf")
plt.close(fig)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'algo':>8}  {'bits':>4}  {'cpu_count':>10}  {'expected':>10}  "
      f"{'ratio':>6}  {'cpu_ms':>8}  {'gpu_ms':>8}  {'speedup':>8}")
for r in rows:
    algo, bits, cpu_cnt, cpu_ms, gpu_batch, gpu_ms, exp = r
    ratio   = cpu_cnt / exp
    speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
    print(f"{algo:>8}  {bits:>4}  {cpu_cnt:>10}  {exp:>10}  "
          f"{ratio:>6.2f}  {cpu_ms:>8.3f}  {gpu_ms:>8.3f}  {speedup:>8.1f}x")
