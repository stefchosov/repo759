# Code generated with assistance from Claude Code (Anthropic CLI)
# Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Usage: Final project — Task 3 OpenMP thread-count scaling plot
#
# Reads: Final/Data/task3/omp_scaling.dat
#   columns: threads  bits  omp_ms
#
# Produces:
#   Final/Data/task3/omp_scaling_time.pdf/.png   — time vs threads (log-log)
#   Final/Data/task3/omp_scaling_speedup.pdf/.png — speedup vs threads vs ideal

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DAT = "Final/Data/task3/omp_scaling.dat"
if not os.path.exists(DAT):
    sys.exit(f"ERROR: missing {DAT}\nRun task3_omp_scaling.sh first.")

# ── Parse ─────────────────────────────────────────────────────────────────────
rows = []
with open(DAT) as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append((int(parts[0]), int(parts[1]), float(parts[2])))

all_bits    = sorted(set(r[1] for r in rows))
all_threads = sorted(set(r[0] for r in rows))

colors = {40: "#4c72b0", 48: "#dd8452"}

def get(bits):
    return {t: ms for t, b, ms in rows if b == bits}

# ── Plot 1: Time vs threads ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for bits in all_bits:
    d = get(bits)
    ts  = sorted(d)
    ms  = [d[t] for t in ts]
    ax.plot(ts, ms, "o-", color=colors.get(bits, "gray"), linewidth=2,
            markersize=7, label=f"{bits}-bit hash")

ax.set_xlabel("OMP_NUM_THREADS", fontsize=12)
ax.set_ylabel("Time to Find Collision (ms)", fontsize=12)
ax.set_title("OpenMP Collision Search: Time vs Thread Count (MD5)", fontsize=12)
ax.set_xticks(all_threads)
ax.set_xticklabels([str(t) for t in all_threads])
ax.set_yscale("log")
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Final/Data/task3/omp_scaling_time.pdf", dpi=150)
fig.savefig("Final/Data/task3/omp_scaling_time.png", dpi=150)
print("Saved Final/Data/task3/omp_scaling_time.pdf")
plt.close(fig)

# ── Plot 2: Speedup vs threads ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for bits in all_bits:
    d = get(bits)
    if 1 not in d:
        # fall back to smallest thread count as baseline
        base_t = min(d)
    else:
        base_t = 1
    base_ms = d[base_t]
    ts      = sorted(d)
    speedups = [base_ms / d[t] for t in ts]
    ax.plot(ts, speedups, "o-", color=colors.get(bits, "gray"), linewidth=2,
            markersize=7, label=f"{bits}-bit hash")

# Ideal linear speedup from thread=1
t_ref = min(all_threads)
t_max = max(all_threads)
ideal_ts = np.linspace(t_ref, t_max, 100)
ax.plot(ideal_ts, ideal_ts / t_ref, "k--", linewidth=1.5, label="Ideal linear")

ax.set_xlabel("OMP_NUM_THREADS", fontsize=12)
ax.set_ylabel("Speedup over 1-thread", fontsize=12)
ax.set_title("OpenMP Speedup vs Thread Count (MD5)", fontsize=12)
ax.set_xticks(all_threads)
ax.set_xticklabels([str(t) for t in all_threads])
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Final/Data/task3/omp_scaling_speedup.pdf", dpi=150)
fig.savefig("Final/Data/task3/omp_scaling_speedup.png", dpi=150)
print("Saved Final/Data/task3/omp_scaling_speedup.pdf")
plt.close(fig)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'threads':>8}  {'bits':>4}  {'omp_ms':>9}  {'speedup':>9}")
for bits in all_bits:
    d = get(bits)
    base_ms = d.get(1, d[min(d)])
    for t in sorted(d):
        print(f"{t:>8}  {bits:>4}  {d[t]:>9.3f}  {base_ms/d[t]:>9.2f}x")
