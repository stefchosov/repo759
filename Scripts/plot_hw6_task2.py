"""
Plot HW06 Task 2 scaling results: Hillis-Steele inclusive scan.
Reads:
  HW06/scaling_task2_n.dat    -- time vs n (fixed tpb=1024)
  HW06/scaling_task2_tpb.dat  -- time vs tpb (fixed n=1048576)
Saves figures to HW06/task2.pdf + task2.png.

Usage (after running the scaling sbatch job):
    python3 Scripts/plot_hw6_task2.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_N   = os.path.join("HW06", "scaling_task2_n.dat")
DATA_TPB = os.path.join("HW06", "scaling_task2_tpb.dat")
OUT_PDF  = os.path.join("HW06", "task2.pdf")
OUT_PNG  = os.path.join("HW06", "task2.png")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data_n   = np.loadtxt(DATA_N,   comments="#")
data_tpb = np.loadtxt(DATA_TPB, comments="#")

n_vals   = data_n[:, 0].astype(int)
time_n   = data_n[:, 1]

tpb_vals = data_tpb[:, 0].astype(int)
time_tpb = data_tpb[:, 1]

# Effective bandwidth (GB/s): each element is read once and written once
BYTES_PER_FLOAT = 4
bw_n = (2.0 * n_vals * BYTES_PER_FLOAT) / (time_n * 1e-3) / 1e9

# ---------------------------------------------------------------------------
# Figure: 2 rows x 2 columns
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# ---- Row 0: vary n ----
ax = axes[0, 0]
ax.loglog(n_vals, time_n, "o-", color="steelblue", linewidth=2, markersize=6)
ax.set_xlabel("n (number of elements)", fontsize=11)
ax.set_ylabel("Time (ms)", fontsize=11)
ax.set_title("Task 2: Time vs n  (tpb = 1024)", fontsize=11)
ax.grid(True, which="both", linestyle="--", alpha=0.5)

ax2 = axes[0, 1]
ax2.semilogx(n_vals, bw_n, "s-", color="steelblue", linewidth=2, markersize=6)
ax2.set_xlabel("n (number of elements)", fontsize=11)
ax2.set_ylabel("Effective bandwidth (GB/s)", fontsize=11)
ax2.set_title("Task 2: Bandwidth vs n  (tpb = 1024)", fontsize=11)
ax2.grid(True, which="both", linestyle="--", alpha=0.5)

# ---- Row 1: vary tpb ----
ax3 = axes[1, 0]
ax3.plot(tpb_vals, time_tpb, "o-", color="darkorange", linewidth=2, markersize=6)
ax3.set_xscale("log", base=2)
ax3.set_xlabel("threads_per_block", fontsize=11)
ax3.set_ylabel("Time (ms)", fontsize=11)
ax3.set_title("Task 2: Time vs tpb  (n = 1 048 576)", fontsize=11)
ax3.set_xticks(tpb_vals)
ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax3.grid(True, which="both", linestyle="--", alpha=0.5)

# Fixed-n bandwidth vs tpb
bw_tpb_n = 1048576
bw_tpb = (2.0 * bw_tpb_n * BYTES_PER_FLOAT) / (time_tpb * 1e-3) / 1e9
ax4 = axes[1, 1]
ax4.plot(tpb_vals, bw_tpb, "s-", color="darkorange", linewidth=2, markersize=6)
ax4.set_xscale("log", base=2)
ax4.set_xlabel("threads_per_block", fontsize=11)
ax4.set_ylabel("Effective bandwidth (GB/s)", fontsize=11)
ax4.set_title("Task 2: Bandwidth vs tpb  (n = 1 048 576)", fontsize=11)
ax4.set_xticks(tpb_vals)
ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax4.grid(True, which="both", linestyle="--", alpha=0.5)

import matplotlib.ticker  # needed for ScalarFormatter used above

fig.suptitle("HW06 Task 2: Hillis-Steele Inclusive Scan", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved {OUT_PDF} and {OUT_PNG}")
