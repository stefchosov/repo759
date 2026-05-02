# Code generated with assistance from Claude Code (Anthropic CLI)
# Model: Claude Sonnet 4.6 (claude-sonnet-4-6)
# Usage: Final project — Task 4 GPU algorithm comparison plots
#
# Reads:
#   Final/Data/task4/scaling_task4_device.dat   — Thrust, pure VRAM (bits ≤ 48)
#   Final/Data/task4/scaling_task4_unified.dat  — Thrust, unified memory (bits ≤ 56)
#   Final/Data/task3/scaling_task3_*.dat        — Pollard's rho GPU times
#
# Produces:
#   task4_compare.pdf   — time vs bits, all three algorithms (one panel per hash)
#   task4_overhead.pdf  — unified/device overhead ratio (linear scale)
#   task4_perround.pdf  — Thrust unified per-round time vs Pollard's rho at large bits
#   task4_speedup.pdf   — Thrust speedup over Pollard's rho

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

# ── Parse ────────────────────────────────────────────────────────────────────
def parse_t4_single(path):
    """Single-trial format: mode algo bits count ms expected
    Returns dict[algo] -> dict[bits] -> (ms, count, expected)"""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            p = line.split()
            if len(p) < 6: continue
            algo, bits = p[1], int(p[2])
            count, ms, exp = int(p[3]), float(p[4]), int(p[5])
            if ms < 0: continue
            out.setdefault(algo, {})[bits] = (ms, count, exp)
    return out

def parse_t4_multi(path):
    """Multi-trial format: trial mode algo bits count ms expected
    Returns dict[algo] -> dict[bits] -> {ms: [list], count: [list], expected: int}.
    Falls back to single-trial format if no leading trial column is present."""
    raw = {}  # (algo, bits) -> list of (ms, count, exp)
    multi = False
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            p = line.split()
            if len(p) >= 7 and p[0].isdigit():
                # trial mode algo bits count ms expected
                multi = True
                algo, bits = p[2], int(p[3])
                count, ms, exp = int(p[4]), float(p[5]), int(p[6])
            elif len(p) >= 6:
                # mode algo bits count ms expected
                algo, bits = p[1], int(p[2])
                count, ms, exp = int(p[3]), float(p[4]), int(p[5])
            else:
                continue
            if ms < 0: continue
            raw.setdefault((algo, bits), []).append((ms, count, exp))

    out = {}
    for (algo, bits), trials in raw.items():
        ms_list    = sorted(t[0] for t in trials)
        cnt_list   = sorted(t[1] for t in trials)
        exp        = trials[0][2]
        out.setdefault(algo, {})[bits] = {
            'ms_list':    ms_list,
            'count_list': cnt_list,
            'expected':   exp,
            'multi':      multi,
        }
    return out

def median(lst):
    if not lst: return None
    n = len(lst)
    return lst[n//2] if n % 2 else 0.5 * (lst[n//2 - 1] + lst[n//2])

def iqr(lst):
    if len(lst) < 2: return (lst[0], lst[0]) if lst else (None, None)
    n = len(lst)
    q1 = lst[n//4]
    q3 = lst[(3*n)//4]
    return (q1, q3)

t4_dev_raw = parse_t4_multi(T4_DEV)   # single trial → 1-element lists
t4_uni_raw = parse_t4_multi(T4_UNI)

# Convenience: dict[algo][bits] -> median ms (matches old API for plotting)
def median_view(raw):
    out = {}
    for algo, byb in raw.items():
        for bits, d in byb.items():
            out.setdefault(algo, {})[bits] = (median(d['ms_list']),
                                              median(d['count_list']),
                                              d['expected'])
    return out

t4_dev = median_view(t4_dev_raw)
t4_uni = median_view(t4_uni_raw)

t3 = {}  # algo -> dict[bits] -> gpu_ms
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
colors = {"md5": "#4c72b0", "sha1": "#dd8452", "sha256": "#55a868"}

# Helper: rounds = count / expected (each round computes batch = mult × expected
# but mult varies by bits; use count/expected as a proxy for "work multiplier")
def rounds_used(algo_data, bits):
    if bits not in algo_data: return None
    _, count, exp = algo_data[bits]
    return count / exp if exp else None

# ── Plot 1: time vs bits, three algorithms, log y ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle("Task 4: Pollard's ρ vs Thrust Sort (device & unified memory)",
             fontsize=12, fontweight="bold")

for ax, algo in zip(axes, algos):
    pol_bits = sorted(t3[algo])
    if pol_bits:
        ax.plot(pol_bits, [t3[algo][b] for b in pol_bits],
                "s--", color="#55a868", linewidth=2, markersize=7,
                label="Pollard's ρ (O(1) memory)")

    dev_bits = [b for b in sorted(t4_dev.get(algo, {})) if b > 16]  # skip warmup
    if dev_bits:
        ax.plot(dev_bits, [t4_dev[algo][b][0] for b in dev_bits],
                "o-", color="#c44e52", linewidth=2, markersize=7,
                label="Thrust sort (VRAM)")

    uni_bits = [b for b in sorted(t4_uni.get(algo, {})) if b > 16]
    if uni_bits:
        # Median + IQR error bars across trials
        med_y = [t4_uni[algo][b][0] for b in uni_bits]
        q1y, q3y = [], []
        for b in uni_bits:
            q1, q3 = iqr(t4_uni_raw[algo][b]['ms_list'])
            q1y.append(q1); q3y.append(q3)
        yerr_lo = [m - q1 for m, q1 in zip(med_y, q1y)]
        yerr_hi = [q3 - m for m, q3 in zip(med_y, q3y)]
        ax.errorbar(uni_bits, med_y, yerr=[yerr_lo, yerr_hi],
                    fmt="^:", color="#8172b3", linewidth=2, markersize=8,
                    capsize=4, label="Thrust sort (unified, median ± IQR)")

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

# ── Plot 2: unified/device overhead ratio (linear scale) ────────────────────
# Makes the 2× constant overhead clearly visible — log-scale plots compress it.
fig, ax = plt.subplots(figsize=(9, 5))

for algo in algos:
    common = sorted(b for b in t4_dev.get(algo, {})
                    if b in t4_uni.get(algo, {}) and b > 16)
    if not common: continue
    ratios = [t4_uni[algo][b][0] / t4_dev[algo][b][0] for b in common]
    ax.plot(common, ratios, "o-", color=colors[algo],
            linewidth=2, markersize=7, label=labels[algo])

ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="No overhead")
ax.axhline(2.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6,
           label="2× page-tracking baseline")
ax.set_xlabel("Truncated Bits", fontsize=12)
ax.set_ylabel("Time(unified) / Time(device)", fontsize=12)
ax.set_title("Unified Memory Overhead vs Device Memory\n"
             "(values > 1 = unified is slower at the same bits)", fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Final/Data/task4/task4_overhead.pdf", dpi=150)
fig.savefig("Final/Data/task4/task4_overhead.png", dpi=150)
print("Saved Final/Data/task4/task4_overhead.pdf")
plt.close(fig)

# ── Plot 3: per-round Thrust unified time vs Pollard's rho at bits=56 ───────
# Total wall time is misleading — Thrust unified at bits=56 uses 1× over-
# allocation and incurs a variable retry count. The fair comparison is
# per-round time vs Pollard's rho's deterministic single-launch time.
fig, ax = plt.subplots(figsize=(8, 5))

x = list(range(len(algos)))
width = 0.30

pol_56  = [t3[a].get(56, 0) for a in algos]
uni_56_total = [t4_uni[a].get(56, (0,0,0))[0] for a in algos]
uni_56_per   = []
for a in algos:
    if 56 in t4_uni.get(a, {}):
        ms, count, exp = t4_uni[a][56]
        rounds = max(1, count / exp)
        uni_56_per.append(ms / rounds)
    else:
        uni_56_per.append(0)

b1 = ax.bar([xi - width for xi in x], pol_56, width,
            color="#55a868", label="Pollard's ρ (single deterministic run)")
b2 = ax.bar(x, uni_56_total, width,
            color="#8172b3", alpha=0.55,
            label="Thrust unified (total, includes retries)")
b3 = ax.bar([xi + width for xi in x], uni_56_per, width,
            color="#8172b3", label="Thrust unified (per-round)")

# Annotate retry counts
for xi, a in zip(x, algos):
    r = rounds_used(t4_uni.get(a, {}), 56)
    if r is not None:
        ax.text(xi, uni_56_total[algos.index(a)] * 1.02,
                f"{r:.0f} rounds", ha="center", fontsize=8, color="#5a4a8a")

ax.set_xticks(x)
ax.set_xticklabels([labels[a] for a in algos])
ax.set_ylabel("Time at bits=56 (ms)", fontsize=12)
ax.set_title("Thrust Unified vs Pollard's ρ at bits=56\n"
             "Per-round Thrust beats Pollard; total time depends on retry luck",
             fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("Final/Data/task4/task4_perround.pdf", dpi=150)
fig.savefig("Final/Data/task4/task4_perround.png", dpi=150)
print("Saved Final/Data/task4/task4_perround.pdf")
plt.close(fig)

# ── Plot 3.5: per-trial spread at bits >= 52 (boxplot or strip) ──────────────
# Visualizes retry-driven variance in unified-mode runs at borderline-VRAM
# bit widths. Only plotted if multi-trial data is available.
high_bits = [52, 56, 64]
have_multi = any(t4_uni_raw.get(a, {}).get(b, {}).get('multi', False)
                 for a in algos for b in high_bits if b in t4_uni_raw.get(a, {}))

if have_multi:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Thrust Unified Memory: Per-trial Spread at High Bit Widths",
                 fontsize=12, fontweight="bold")

    for ax, algo in zip(axes, algos):
        positions = []
        data = []
        labels_x = []
        for b in high_bits:
            if b not in t4_uni_raw.get(algo, {}): continue
            ms_list = t4_uni_raw[algo][b]['ms_list']
            if len(ms_list) < 2: continue
            positions.append(len(positions) + 1)
            data.append(ms_list)
            labels_x.append(f"bits={b}\n(n={len(ms_list)})")

        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.5,
                            patch_artist=True, showfliers=True)
            for patch in bp['boxes']:
                patch.set_facecolor("#8172b3")
                patch.set_alpha(0.5)
            # Overlay Pollard's rho horizontal lines for comparison
            for i, b in enumerate([b for b in high_bits if b in t4_uni_raw.get(algo, {})]):
                if b in t3[algo] and len(t4_uni_raw[algo][b]['ms_list']) >= 2:
                    ax.hlines(t3[algo][b], positions[i] - 0.3, positions[i] + 0.3,
                              colors="#55a868", linestyles="--", linewidth=2)

        ax.set_title(labels[algo], fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_x, fontsize=9)
        ax.set_yscale("log")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        if not positions:
            ax.text(0.5, 0.5, "no multi-trial data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")

    axes[0].set_ylabel("Time per run (ms)", fontsize=10)
    # Single legend annotation
    from matplotlib.lines import Line2D
    custom = [Line2D([0],[0], color="#8172b3", lw=8, alpha=0.5, label="Thrust unified (10 trials)"),
              Line2D([0],[0], color="#55a868", linestyle="--", lw=2, label="Pollard's ρ (deterministic)")]
    fig.legend(handles=custom, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig("Final/Data/task4/task4_trial_spread.pdf", dpi=150)
    fig.savefig("Final/Data/task4/task4_trial_spread.png", dpi=150)
    print("Saved Final/Data/task4/task4_trial_spread.pdf")
    plt.close(fig)

# ── Plot 4: Thrust speedup over Pollard's ρ (log y) ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for algo in algos:
    common_dev = sorted(b for b in t4_dev.get(algo, {}) if b in t3[algo] and b > 16)
    if common_dev:
        ax.plot(common_dev, [t3[algo][b] / t4_dev[algo][b][0] for b in common_dev],
                "o-", color=colors[algo], linewidth=2, markersize=7,
                label=f"{labels[algo]} — Thrust device")

    common_uni = sorted(b for b in t4_uni.get(algo, {}) if b in t3[algo] and b > 16)
    if common_uni:
        ax.plot(common_uni, [t3[algo][b] / t4_uni[algo][b][0] for b in common_uni],
                "^:", color=colors[algo], linewidth=2, markersize=8, alpha=0.65,
                label=f"{labels[algo]} — Thrust unified")

ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Equal performance")
ax.set_xlabel("Truncated Bits", fontsize=12)
ax.set_ylabel("Pollard's ρ time / Thrust time (speedup)", fontsize=12)
ax.set_title("Thrust Sort Speedup over Pollard's ρ\n"
             "(>1 = Thrust faster; <1 = Pollard's ρ faster)", fontsize=11)
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
print(f"\n{'algo':>8}  {'bits':>4}  {'pollard_ms':>11}  {'dev_ms':>9}  "
      f"{'uni_ms':>9}  {'uni_rounds':>10}  {'uni_per_round':>13}  "
      f"{'uni/dev':>8}  {'pol/uni':>8}")
for algo in algos:
    all_b = sorted(set(t3[algo]) | set(t4_dev.get(algo,{})) | set(t4_uni.get(algo,{})))
    for b in all_b:
        pol = t3[algo].get(b)
        dev = t4_dev.get(algo, {}).get(b)
        uni = t4_uni.get(algo, {}).get(b)
        pol_s   = f"{pol:.3f}"          if pol is not None else "—"
        dev_s   = f"{dev[0]:.3f}"       if dev else "—"
        uni_s   = f"{uni[0]:.3f}"       if uni else "—"
        rnd_s   = f"{uni[1]/uni[2]:.1f}" if uni else "—"
        per_s   = f"{uni[0] / max(1, uni[1]/uni[2]):.3f}" if uni else "—"
        ud_s    = f"{uni[0]/dev[0]:.2f}x" if (uni and dev) else "—"
        pu_s    = f"{pol/uni[0]:.2f}x"  if (pol and uni) else "—"
        print(f"{algo:>8}  {b:>4}  {pol_s:>11}  {dev_s:>9}  "
              f"{uni_s:>9}  {rnd_s:>10}  {per_s:>13}  {ud_s:>8}  {pu_s:>8}")
