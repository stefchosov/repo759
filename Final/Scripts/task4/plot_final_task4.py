# Code generated with assistance from Claude Code (Anthropic CLI)
# Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Usage: Final project — Task 4 GPU algorithm comparison
#
# Reads:
#   Final/Data/task4/scaling_task4_device.dat   (Thrust, pure VRAM)
#   Final/Data/task4/scaling_task4_unified.dat  (Thrust, unified memory)
#   Final/Data/task3/scaling_task3_*.dat        (Pollard's rho, for comparison)
#
# Produces:
#   Final/Data/task4/task4_compare.pdf/.png  — Pollard vs Thrust-dev vs Thrust-unified
#   Final/Data/task4/task4_speedup.pdf/.png  — Thrust speedup over Pollard's rho

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T4_DEV = "Final/Data/task4/scaling_task4_device.dat"
T4_UNI = "Final/Data/task4/scaling_task4_unified.dat"
T3 = {
    "md5":    "Final/Data/task3/scaling_task3_md5.dat",
    "sha1":   "Final/Data/task3/scaling_task3_sha1.dat",
    "sha256": "Final/Data/task3/scaling_task3_sha256.dat",
}

for p in [T4_DEV, T4_UNI] + list(T3.values()):
    if not os.path.exists(p):
        sys.exit(f"ERROR: missing {p}")

# ── Parse a Task 4 file ──────────────────────────────────────────────────────
def parse_t4(path):
    """Returns dict[algo] -> dict[bits] -> ms"""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            p = line.split()
            if len(p) < 6: continue
            # mode algo bits count ms expected
            algo, bits, ms = p[1], int(p[2]), float(p[4])
            if ms < 0: continue  # OOM / failure
            out.setdefault(algo, {})[bits] = ms
    return out

t4_dev = parse_t4(T4_DEV)
t4_uni = parse_t4(T4_UNI)

# ── Parse Task 3 GPU column ──────────────────────────────────────────────────
t3 = {}
for algo, path in T3.items():
    t3[algo] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            p = line.split()
            if len(p) < 9: continue
            t3[algo][int(p[1])] = float(p[7])

algos  = ["md5", "sha1", "sha256"]
labels = {"md5": "MD5", "sha1": "SHA-1", "sha256": "SHA-256"}

# ── Plot 1: time vs bits — three algorithms, one panel per hash function ────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle("Task 4: GPU Algorithm Comparison — Pollard's ρ vs Thrust Sort (device & unified memory)",
             fontsize=12, fontweight="bold")

for ax, algo in zip(axes, algos):
    pol_bits = sorted(t3[algo])
    if pol_bits:
        ax.plot(pol_bits, [t3[algo][b] for b in pol_bits],
                "s--", color="#55a868", linewidth=2, markersize=7,
                label="Pollard's ρ (O(1) memory)")

    dev_bits = sorted(t4_dev.get(algo, {}))
    if dev_bits:
        ax.plot(dev_bits, [t4_dev[algo][b] for b in dev_bits],
                "o-", color="#c44e52", linewidth=2, markersize=7,
                label="Thrust sort (VRAM)")

    uni_bits = sorted(t4_uni.get(algo, {}))
    if uni_bits:
        ax.plot(uni_bits, [t4_uni[algo][b] for b in uni_bits],
                "^:", color="#8172b3", linewidth=2, markersize=8,
                label="Thrust sort (unified memory)")

    ax.set_title(labels[algo], fontsize=12)
    ax.set_yscale("log")
    all_bits = sorted(set(pol_bits) | set(dev_bits) | set(uni_bits))
    ax.set_xticks(all_bits)
    ax.set_xticklabels([str(b) for b in all_bits], rotation=45)
    ax.set_xlabel("Truncated Bits", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Time (ms)", fontsize=10)
fig.tight_layout()
fig.savefig("Final/Data/task4/task4_compare.pdf", dpi=150)
fig.savefig("Final/Data/task4/task4_compare.png", dpi=150)
print("Saved Final/Data/task4/task4_compare.pdf")
plt.close(fig)

# ── Plot 2: Thrust speedup over Pollard's ρ (both modes) ────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"md5": "#4c72b0", "sha1": "#dd8452", "sha256": "#55a868"}

for algo in algos:
    common_dev = sorted(b for b in t4_dev.get(algo, {}) if b in t3[algo])
    if common_dev:
        ax.plot(common_dev, [t3[algo][b] / t4_dev[algo][b] for b in common_dev],
                "o-", color=colors[algo], linewidth=2, markersize=7,
                label=f"{labels[algo]} — Thrust (VRAM)")

    common_uni = sorted(b for b in t4_uni.get(algo, {}) if b in t3[algo])
    if common_uni:
        ax.plot(common_uni, [t3[algo][b] / t4_uni[algo][b] for b in common_uni],
                "^:", color=colors[algo], linewidth=2, markersize=8, alpha=0.65,
                label=f"{labels[algo]} — Thrust (unified)")

ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
           label="Equal performance")
ax.set_xlabel("Truncated Bits", fontsize=12)
ax.set_ylabel("Pollard's ρ time / Thrust time (speedup)", fontsize=12)
ax.set_title("Thrust Sort Speedup over Pollard's ρ\n"
             "(values > 1 = Thrust faster; values < 1 = Pollard's ρ faster)",
             fontsize=11)
ax.set_yscale("log")
ax.legend(fontsize=8, ncol=2)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Final/Data/task4/task4_speedup.pdf", dpi=150)
fig.savefig("Final/Data/task4/task4_speedup.png", dpi=150)
print("Saved Final/Data/task4/task4_speedup.pdf")
plt.close(fig)

# ── Summary table ────────────────────────────────────────────────────────────
print(f"\n{'algo':>8}  {'bits':>4}  {'pollard_ms':>11}  {'thrust_dev':>11}  {'thrust_uni':>11}  "
      f"{'dev_spd':>9}  {'uni_spd':>9}")
for algo in algos:
    all_b = sorted(set(t3[algo]) | set(t4_dev.get(algo,{})) | set(t4_uni.get(algo,{})))
    for b in all_b:
        pol = t3[algo].get(b)
        dev = t4_dev.get(algo, {}).get(b)
        uni = t4_uni.get(algo, {}).get(b)
        pol_s   = f"{pol:.3f}"     if pol is not None else "—"
        dev_s   = f"{dev:.3f}"     if dev is not None else "—"
        uni_s   = f"{uni:.3f}"     if uni is not None else "—"
        dev_spd = f"{pol/dev:.2f}x" if (pol and dev) else "—"
        uni_spd = f"{pol/uni:.2f}x" if (pol and uni) else "—"
        print(f"{algo:>8}  {b:>4}  {pol_s:>11}  {dev_s:>11}  {uni_s:>11}  "
              f"{dev_spd:>9}  {uni_spd:>9}")
