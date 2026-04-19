# Code generated with assistance from Claude Code (Anthropic CLI)
# Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Usage: Final project — Task 3 collision search plots
#
# Reads: Final/Data/task3/scaling_task3_{md5,sha1,sha256}.dat
#   columns: algo  bits  cpu_count  cpu_ms  omp_count  omp_ms  gpu_count  gpu_ms  expected
#   cpu_count / cpu_ms / omp_count / omp_ms = -1 when bits > CPU_MAX_BITS
#
# Produces (one subplot per algorithm in each figure):
#   Final/Data/task3/task3_time.pdf/.png
#   Final/Data/task3/task3_birthday.pdf/.png
#   Final/Data/task3/task3_speedup.pdf/.png

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

# ── Parse ─────────────────────────────────────────────────────────────────────
rows = []
for algo, path in DAT_FILES.items():
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            # bits=72 and 80 have -1 for cpu/omp columns — that's fine, handled below
            rows.append((
                parts[0],        # algo
                int(parts[1]),   # bits
                int(parts[2]),   # cpu_cnt
                float(parts[3]), # cpu_ms
                int(parts[4]),   # omp_cnt
                float(parts[5]), # omp_ms
                int(parts[6]),   # gpu_cnt
                float(parts[7]), # gpu_ms
                int(parts[8]),   # expected
            ))

algos  = ["md5", "sha1", "sha256"]
labels = {"md5": "MD5", "sha1": "SHA-1", "sha256": "SHA-256"}
all_bits = sorted(set(r[1] for r in rows))

def algo_rows(algo):
    return [r for r in rows if r[0] == algo]

def valid(algo, ms_idx):
    return [(r[1], r[ms_idx]) for r in algo_rows(algo) if r[ms_idx] >= 0]

# ── Plot 1: Time vs bits — one subplot per algorithm ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig.suptitle("Truncated Collision Search: Time vs Bit Width", fontsize=13, fontweight="bold")

for ax, algo in zip(axes, algos):
    cpu_pts = valid(algo, 3)
    omp_pts = valid(algo, 5)
    gpu_pts = [(r[1], r[7]) for r in algo_rows(algo)]

    if cpu_pts:
        ax.plot([p[0] for p in cpu_pts], [p[1] for p in cpu_pts],
                "o-", color="#4c72b0", linewidth=2, markersize=6, label="CPU serial")
    if omp_pts:
        ax.plot([p[0] for p in omp_pts], [p[1] for p in omp_pts],
                "^--", color="#dd8452", linewidth=2, markersize=6, label="OMP (12 threads)")
    ax.plot([p[0] for p in gpu_pts], [p[1] for p in gpu_pts],
            "s-.", color="#55a868", linewidth=2, markersize=6, label="GPU (Pollard's ρ)")

    ax.set_title(labels[algo], fontsize=12)
    ax.set_yscale("log")
    ax.set_xticks(all_bits)
    ax.set_xticklabels([str(b) for b in all_bits], rotation=45)
    ax.set_xlabel("Truncated Bits", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Time to Find Collision (ms)", fontsize=10)
fig.tight_layout()
fig.savefig("Final/Data/task3/task3_time.pdf", dpi=150)
fig.savefig("Final/Data/task3/task3_time.png", dpi=150)
print("Saved Final/Data/task3/task3_time.pdf")
plt.close(fig)

# ── Plot 2: Birthday paradox validation — one subplot per algorithm ───────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig.suptitle("Birthday Paradox Validation: Actual CPU Count / Expected Count",
             fontsize=13, fontweight="bold")

for ax, algo in zip(axes, algos):
    pts     = {r[1]: r[2] for r in algo_rows(algo) if r[2] >= 0}
    exp_pts = {r[1]: r[8] for r in algo_rows(algo)}
    cpu_bits = sorted(pts)
    ratios   = [pts[b] / exp_pts[b] for b in cpu_bits]

    ax.bar(range(len(cpu_bits)), ratios, color="#4c72b0", alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Theoretical = 1.0")
    ax.set_title(labels[algo], fontsize=12)
    ax.set_xticks(range(len(cpu_bits)))
    ax.set_xticklabels([str(b) for b in cpu_bits], rotation=45)
    ax.set_xlabel("Truncated Bits", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Actual / Expected", fontsize=10)
fig.tight_layout()
fig.savefig("Final/Data/task3/task3_birthday.pdf", dpi=150)
fig.savefig("Final/Data/task3/task3_birthday.png", dpi=150)
print("Saved Final/Data/task3/task3_birthday.pdf")
plt.close(fig)

# ── Plot 3: Speedup over CPU serial — one subplot per algorithm ───────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig.suptitle("Speedup over CPU Serial: OMP (12 threads) vs GPU (Pollard's ρ)",
             fontsize=13, fontweight="bold")

for ax, algo in zip(axes, algos):
    cpu_t = {r[1]: r[3] for r in algo_rows(algo) if r[3] >= 0}
    omp_t = {r[1]: r[5] for r in algo_rows(algo) if r[5] >= 0}
    gpu_t = {r[1]: r[7] for r in algo_rows(algo)}

    common_omp = sorted(set(cpu_t) & set(omp_t))
    common_gpu = sorted(set(cpu_t) & set(gpu_t))

    if common_omp:
        ax.plot(common_omp, [cpu_t[b] / omp_t[b] for b in common_omp],
                "^--", color="#dd8452", linewidth=2, markersize=6, label="OMP (12 threads)")
    if common_gpu:
        ax.plot(common_gpu, [cpu_t[b] / gpu_t[b] for b in common_gpu],
                "s-.", color="#55a868", linewidth=2, markersize=6, label="GPU")

    ax.axhline(1.0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax.set_title(labels[algo], fontsize=12)
    ax.set_yscale("log")
    ax.set_xticks(sorted(set(common_omp) | set(common_gpu)))
    ax.set_xticklabels([str(b) for b in sorted(set(common_omp) | set(common_gpu))], rotation=45)
    ax.set_xlabel("Truncated Bits", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Speedup (CPU time / impl time)", fontsize=10)
fig.tight_layout()
fig.savefig("Final/Data/task3/task3_speedup.pdf", dpi=150)
fig.savefig("Final/Data/task3/task3_speedup.png", dpi=150)
print("Saved Final/Data/task3/task3_speedup.pdf")
plt.close(fig)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'algo':>8}  {'bits':>4}  {'cpu_cnt':>12}  {'omp_cnt':>12}  {'gpu_cnt':>12}  "
      f"{'expected':>12}  {'cpu_ms':>9}  {'omp_ms':>9}  {'gpu_ms':>9}  "
      f"{'omp_spd':>9}  {'gpu_spd':>9}")
for r in rows:
    algo, bits, cpu_cnt, cpu_ms, omp_cnt, omp_ms, gpu_cnt, gpu_ms, expected = r
    omp_spd = cpu_ms / omp_ms if cpu_ms >= 0 and omp_ms > 0 else float("nan")
    gpu_spd = cpu_ms / gpu_ms if cpu_ms >= 0 and gpu_ms > 0 else float("nan")
    print(f"{algo:>8}  {bits:>4}  {cpu_cnt:>12}  {omp_cnt:>12}  {gpu_cnt:>12}  "
          f"{expected:>12}  {cpu_ms:>9.3f}  {omp_ms:>9.3f}  {gpu_ms:>9.3f}  "
          f"{omp_spd:>9.1f}x  {gpu_spd:>9.1f}x")
