# Code generated with assistance from Claude Code (Anthropic CLI)
# Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Usage: Final project — Task 3 collision search plots
#
# Reads: Final/Data/task3/scaling_task3_{md5,sha1,sha256}.dat
#   columns: algo  bits  cpu_count  cpu_ms  gpu_ms  expected
#   cpu_count / cpu_ms = -1 when bits > 56 (CPU not run)
#
# Produces:
#   Final/Data/task3/task3_time.pdf/.png
#     — time (ms) vs truncated bits, log scale, CPU vs GPU per algorithm
#   Final/Data/task3/task3_birthday.pdf/.png
#     — actual CPU count / theoretical expected (birthday paradox validation)
#   Final/Data/task3/task3_speedup.pdf/.png
#     — GPU speedup over CPU vs bits

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DAT_FILES = {
    "md5":    "Final/Data/task3/scaling_task3_md5.dat",
    "sha1":   "Final/Data/task3/scaling_task3_sha1.dat",
    "sha256": "Final/Data/task3/scaling_task3_sha256.dat",
}

missing = [p for p in DAT_FILES.values() if not os.path.exists(p)]
if missing:
    sys.exit(f"ERROR: missing data files: {missing}\nRun the per-algo sbatch scripts first.")

# ── Parse data ────────────────────────────────────────────────────────────────
rows = []
for algo, path in DAT_FILES.items():
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            a        = parts[0]
            bits     = int(parts[1])
            cpu_cnt  = int(parts[2])    # -1 if not run
            cpu_ms   = float(parts[3])  # -1.0 if not run
            gpu_cnt  = int(parts[4])
            gpu_ms   = float(parts[5])
            expected = int(parts[6])
            rows.append((a, bits, cpu_cnt, cpu_ms, gpu_cnt, gpu_ms, expected))

algos  = ["md5", "sha1", "sha256"]
labels = {"md5": "MD5", "sha1": "SHA-1", "sha256": "SHA-256"}
colors = {"md5": "#4c72b0", "sha1": "#dd8452", "sha256": "#55a868"}

def get(algo, field):
    idx = {"bits":1,"cpu_cnt":2,"cpu_ms":3,"gpu_cnt":4,"gpu_ms":5,"expected":6}[field]
    return [(r[1], r[idx]) for r in rows if r[0] == algo]

def valid_cpu(algo):
    return [(b, ms) for b, ms in get(algo, "cpu_ms") if ms >= 0]

# ── Plot 1: Time vs bits ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

for algo in algos:
    c   = colors[algo]
    lbl = labels[algo]

    # GPU — all bits
    pts = get(algo, "gpu_ms")
    bx  = [p[0] for p in pts]
    by  = [p[1] for p in pts]
    ax.plot(bx, by, "s--", color=c, linewidth=2, markersize=7, alpha=0.80,
            label=f"{lbl} GPU (Pollard's ρ)")

    # CPU — bits ≤ 56
    pts = valid_cpu(algo)
    if pts:
        bx = [p[0] for p in pts]
        by = [p[1] for p in pts]
        ax.plot(bx, by, "o-", color=c, linewidth=2, markersize=7,
                label=f"{lbl} CPU")

all_bits = sorted(set(r[1] for r in rows))
ax.set_yscale("log")
ax.set_xticks(all_bits)
ax.set_xticklabels([str(b) for b in all_bits])
ax.set_xlabel("Truncated Hash Bits", fontsize=12)
ax.set_ylabel("Time to Find Collision (ms)", fontsize=12)
ax.set_title("Truncated Collision Search: CPU vs GPU (Pollard's ρ)\n"
             "(solid=CPU sequential, dashed=GPU Pollard's rho)", fontsize=12)
ax.legend(fontsize=8, ncol=2)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Final/Data/task3/task3_time.pdf", dpi=150)
fig.savefig("Final/Data/task3/task3_time.png", dpi=150)
print("Saved Final/Data/task3/task3_time.pdf")
plt.close(fig)

# ── Plot 2: Birthday paradox validation (CPU count / expected) ────────────────
fig, ax = plt.subplots(figsize=(8, 5))

cpu_bits = sorted(set(b for b, _ in valid_cpu("md5")))
x_pos    = np.arange(len(cpu_bits))
width    = 0.25

for di, algo in enumerate(algos):
    pts      = {b: cnt for b, cnt in get(algo, "cpu_cnt") if cnt >= 0}
    exp_pts  = {b: e   for b, e   in get(algo, "expected")}
    ratios   = [pts[b] / exp_pts[b] for b in cpu_bits if b in pts]
    valid_x  = [x_pos[i] for i, b in enumerate(cpu_bits) if b in pts]
    ax.bar([v + (di - 1) * width for v in valid_x], ratios, width,
           label=labels[algo], color=colors[algo], alpha=0.85)

ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Theoretical (= 1.0)")
ax.set_xlabel("Truncated Hash Bits", fontsize=12)
ax.set_ylabel("Actual Count / Expected Count", fontsize=12)
ax.set_title("Birthday Paradox Validation\n"
             "actual CPU collision count vs. √(π/2)·2^(bits/2)", fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels([str(b) for b in cpu_bits])
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Final/Data/task3/task3_birthday.pdf", dpi=150)
fig.savefig("Final/Data/task3/task3_birthday.png", dpi=150)
print("Saved Final/Data/task3/task3_birthday.pdf")
plt.close(fig)

# ── Plot 3: GPU speedup over CPU ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for algo in algos:
    cpu_t = {b: ms for b, ms in valid_cpu(algo)}
    gpu_t = {b: ms for b, ms in get(algo, "gpu_ms")}
    common_bits = sorted(set(cpu_t) & set(gpu_t))
    if not common_bits:
        continue
    speedups = [cpu_t[b] / gpu_t[b] for b in common_bits]
    ax.plot(common_bits, speedups, "o-", color=colors[algo], linewidth=2,
            markersize=7, label=labels[algo])

ax.set_yscale("log")
ax.set_xlabel("Truncated Hash Bits", fontsize=12)
ax.set_ylabel("CPU time / GPU time  (speedup)", fontsize=12)
ax.set_title("GPU Speedup over CPU Sequential Search", fontsize=13)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Final/Data/task3/task3_speedup.pdf", dpi=150)
fig.savefig("Final/Data/task3/task3_speedup.png", dpi=150)
print("Saved Final/Data/task3/task3_speedup.pdf")
plt.close(fig)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'algo':>8}  {'bits':>4}  {'cpu_cnt':>12}  {'gpu_cnt':>12}  {'expected':>12}  "
      f"{'ratio':>6}  {'cpu_ms':>9}  {'gpu_ms':>9}  {'speedup':>9}")
for r in rows:
    algo, bits, cpu_cnt, cpu_ms, gpu_cnt, gpu_ms, expected = r
    ratio   = cpu_cnt / expected if cpu_cnt >= 0 else float("nan")
    speedup = cpu_ms  / gpu_ms   if cpu_ms  >= 0 and gpu_ms > 0 else float("nan")
    print(f"{algo:>8}  {bits:>4}  {cpu_cnt:>12}  {gpu_cnt:>12}  {expected:>12}  "
          f"{ratio:>6.2f}  {cpu_ms:>9.3f}  {gpu_ms:>9.3f}  {speedup:>9.1f}x")
