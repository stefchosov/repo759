"""
Plot HW07 Task 2: count() time vs n in log-log scale.
Reads:  HW07/scaling_task2.dat  (columns: n  time_ms)
Saves:  HW07/task2.pdf  HW07/task2.png

Usage (from repo root):
    python3 Scripts/plot_hw7_task2.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

DATA    = os.path.join("HW07", "scaling_task2.dat")
OUT_PDF = os.path.join("HW07", "task2.pdf")
OUT_PNG = os.path.join("HW07", "task2.png")

data    = np.loadtxt(DATA, comments="#")
n_vals  = data[:, 0].astype(int)
time_ms = data[:, 1]

fig, ax = plt.subplots(figsize=(7, 5))

ax.loglog(n_vals, time_ms, "o-", color="steelblue", linewidth=2, markersize=6)
ax.set_xlabel("n (number of elements)", fontsize=12)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("HW07 Task 2: Thrust count() Time vs n", fontsize=12)
ax.grid(True, which="both", linestyle="--", alpha=0.5)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved {OUT_PDF} and {OUT_PNG}")
