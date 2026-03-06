"""
Plot HW06 Task 2 scaling results: Hillis-Steele inclusive scan time vs n.
Reads HW06/scaling_task2.dat and saves the figure to HW06/task2.pdf + task2.png.

Usage (from repo root after running the scaling sbatch job):
    python3 Scripts/plot_hw6_task2.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

DATA  = os.path.join("HW06", "scaling_task2.dat")
OUT_PDF = os.path.join("HW06", "task2.pdf")
OUT_PNG = os.path.join("HW06", "task2.png")

data    = np.loadtxt(DATA, comments="#")
n_vals  = data[:, 0].astype(int)
time_ms = data[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ---- Left: time vs n ----
ax = axes[0]
ax.plot(n_vals, time_ms, "o-", color="steelblue", linewidth=2, markersize=6)
ax.set_xscale("log", base=2)
ax.set_xlabel("n (number of elements)", fontsize=12)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("HW06 Task 2: Scan Time vs n  (tpb = 1024)", fontsize=11)
ax.set_xticks(n_vals)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.tick_params(axis='x', rotation=45)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.annotate("GPU warmup", xy=(n_vals[0], time_ms[0]),
            xytext=(n_vals[1], time_ms[0] * 0.7),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="gray")

# ---- Right: effective bandwidth vs n ----
BYTES_PER_FLOAT = 4
bw = (2.0 * n_vals * BYTES_PER_FLOAT) / (time_ms * 1e-3) / 1e9

ax2 = axes[1]
ax2.plot(n_vals, bw, "s-", color="darkorange", linewidth=2, markersize=6)
ax2.set_xscale("log", base=2)
ax2.set_xlabel("n (number of elements)", fontsize=12)
ax2.set_ylabel("Effective bandwidth (GB/s)", fontsize=12)
ax2.set_title("HW06 Task 2: Bandwidth vs n  (tpb = 1024)", fontsize=11)
ax2.set_xticks(n_vals)
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, which="both", linestyle="--", alpha=0.5)

import matplotlib.ticker

fig.suptitle("HW06 Task 2: Hillis-Steele Inclusive Scan", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved {OUT_PDF} and {OUT_PNG}")
