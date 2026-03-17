"""
Plot HW08 Task 3: msort time vs thread count t (linear-linear scale).
Reads:  HW08/scaling_task3_t.dat  (columns: t  time_ms)
Saves:  HW08/hw8_task3_t.pdf  HW08/hw8_task3_t.png

Usage (from repo root):
    python3 Scripts/plot_hw8_task3_t.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

DATA    = os.path.join("HW08", "scaling_task3_t.dat")
OUT_PDF = os.path.join("HW08", "hw8_task3_t.pdf")
OUT_PNG = os.path.join("HW08", "hw8_task3_t.png")

data    = np.loadtxt(DATA, comments="#")
t_vals  = data[:, 0].astype(int)
time_ms = data[:, 1]

ideal = time_ms[0] / t_vals

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(t_vals, time_ms, "o-", color="steelblue", linewidth=2, markersize=6, label="Measured")
ax.plot(t_vals, ideal,   "--", color="gray",       linewidth=1.5,            label="Ideal speedup")

ax.set_xlabel("Number of threads (t)", fontsize=12)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("HW08 Task 3: msort Time vs Threads  (n=10⁶)", fontsize=12)
ax.set_xticks(t_vals)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved {OUT_PDF} and {OUT_PNG}")
