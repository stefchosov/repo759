# Code generated with assistance from Claude Code (Anthropic CLI)
# Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Usage: Final project — Task 4 GPU algorithm comparison
#
# Reads:
#   Final/Data/task4/scaling_task4.dat   (Thrust)
#   Final/Data/task3/scaling_task3_*.dat (Pollard's rho — for cross-comparison)
#
# Produces:
#   Final/Data/task4/task4_compare.pdf/.png  — Thrust vs Pollard's time vs bits per algo
#   Final/Data/task4/task4_speedup.pdf/.png  — Thrust speedup over Pollard's rho

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T4 = "Final/Data/task4/scaling_task4.dat"
T3 = {
    "md5":    "Final/Data/task3/scaling_task3_md5.dat",
    "sha1":   "Final/Data/task3/scaling_task3_sha1.dat",
    "sha256": "Final/Data/task3/scaling_task3_sha256.dat",
}

if not os.path.exists(T4):
    sys.exit(f"ERROR: missing {T4}\nRun task4_scaling.sh first.")
for p in T3.values():
    if not os.path.exists(p):
        sys.exit(f"ERROR: missing {p}\nRun the Task 3 sbatch scripts first.")

# ── Parse Task 4 (Thrust) ─────────────────────────────────────────────────────
# columns: algo bits thrust_count thrust_ms expected
t4 = {}  # algo -> {bits: ms}
with open(T4) as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 5: continue
        algo, bits, ms = p[0], int(p[1]), float(p[3])
        t4.setdefault(algo, {})[bits] = ms

# ── Parse Task 3 (Pollard) GPU column ────────────────────────────────────────
# columns: algo bits cpu_count cpu_ms omp_count omp_ms gpu_count gpu_ms expected
t3 = {}  # algo -> {bits: gpu_ms}
for algo, path in T3.items():
    t3[algo] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 9: continue
            t3[algo][int(p[1])] = float(p[7])

algos  = ["md5", "sha1", "sha256"]
labels = {"md5": "MD5", "sha1": "SHA-1", "sha256": "SHA-256"}

# ── Plot 1: time vs bits — Thrust vs Pollard, one panel per algo ─────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig.suptitle("Task 4: GPU Thrust Sort vs Task 3 Pollard's ρ — Time vs Bits",
             fontsize=13, fontweight="bold")

for ax, algo in zip(axes, algos):
    bits_t4 = sorted(t4.get(algo, {}))
    if bits_t4:
        ax.plot(bits_t4, [t4[algo][b] for b in bits_t4],
                "o-", color="#c44e52", linewidth=2, markersize=7, label="Thrust sort")
    bits_t3 = sorted(b for b in t3[algo] if b in t4.get(algo, {}))
    if bits_t3:
        ax.plot(bits_t3, [t3[algo][b] for b in bits_t3],
                "s--", color="#55a868", linewidth=2, markersize=7, label="Pollard's ρ")
    ax.set_title(labels[algo], fontsize=12)
    ax.set_yscale("log")
    all_bits = sorted(set(bits_t4) | set(bits_t3))
    ax.set_xticks(all_bits)
    ax.set_xticklabels([str(b) for b in all_bits], rotation=45)
    ax.set_xlabel("Truncated Bits", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

axes[0].set_ylabel("Time (ms)", fontsize=10)
fig.tight_layout()
fig.savefig("Final/Data/task4/task4_compare.pdf", dpi=150)
fig.savefig("Final/Data/task4/task4_compare.png", dpi=150)
print("Saved Final/Data/task4/task4_compare.pdf")
plt.close(fig)

# ── Plot 2: Thrust speedup over Pollard ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
colors = {"md5": "#4c72b0", "sha1": "#dd8452", "sha256": "#55a868"}

for algo in algos:
    common = sorted(b for b in t4.get(algo, {}) if b in t3[algo])
    if not common: continue
    speedups = [t3[algo][b] / t4[algo][b] for b in common]
    ax.plot(common, speedups, "o-", color=colors[algo], linewidth=2,
            markersize=7, label=labels[algo])

ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
           label="Equal performance (= 1.0)")
ax.set_xlabel("Truncated Bits", fontsize=12)
ax.set_ylabel("Pollard's ρ time / Thrust time  (speedup)", fontsize=12)
ax.set_title("Thrust Sort Speedup over Pollard's ρ\n"
             "(values > 1 = Thrust is faster)", fontsize=12)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Final/Data/task4/task4_speedup.pdf", dpi=150)
fig.savefig("Final/Data/task4/task4_speedup.png", dpi=150)
print("Saved Final/Data/task4/task4_speedup.pdf")
plt.close(fig)

# ── Summary table ────────────────────────────────────────────────────────────
print(f"\n{'algo':>8}  {'bits':>4}  {'thrust_ms':>10}  {'pollard_ms':>11}  {'thrust_speedup':>15}")
for algo in algos:
    for b in sorted(set(t4.get(algo, {})) | set(t3[algo])):
        tms = t4.get(algo, {}).get(b)
        pms = t3[algo].get(b)
        if tms is None or pms is None:
            print(f"{algo:>8}  {b:>4}  {tms if tms is not None else '—':>10}  "
                  f"{pms if pms is not None else '—':>11}  {'—':>15}")
        else:
            print(f"{algo:>8}  {b:>4}  {tms:>10.3f}  {pms:>11.3f}  {pms/tms:>14.2f}x")
