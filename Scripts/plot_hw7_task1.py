"""
Plot HW07 Task 1: overlay Thrust, CUB, and HW05 reduce on one log-log plot.
Reads:
  HW07/scaling_task1_thrust.dat  -- columns: n  time_ms
  HW07/scaling_task1_cub.dat     -- columns: n  time_ms
  HW05/scaling_task2.dat         -- columns: N  time_tpb1024_ms  time_tpb256_ms
Saves: HW07/task1.pdf  HW07/task1.png

Usage (from repo root):
    python3 Scripts/plot_hw7_task1.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

THRUST = os.path.join("HW07", "scaling_task1_thrust.dat")
CUB    = os.path.join("HW07", "scaling_task1_cub.dat")
HW05   = os.path.join("HW05", "scaling_task2.dat")
OUT_PDF = os.path.join("HW07", "task1.pdf")
OUT_PNG = os.path.join("HW07", "task1.png")

d_thrust = np.loadtxt(THRUST, comments="#")
d_cub    = np.loadtxt(CUB,    comments="#")
d_hw05   = np.loadtxt(HW05,   comments="#")

n_thrust  = d_thrust[:, 0].astype(int)
t_thrust  = d_thrust[:, 1]

n_cub     = d_cub[:, 0].astype(int)
t_cub     = d_cub[:, 1]

n_hw05    = d_hw05[:, 0].astype(int)
t_hw05    = d_hw05[:, 1]   # tpb=1024 column

fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(n_hw05,   t_hw05,   "o--", color="gray",       linewidth=2, markersize=6, label="HW05 reduce (tpb=1024)")
ax.loglog(n_thrust, t_thrust, "s-",  color="steelblue",  linewidth=2, markersize=6, label="Thrust reduce")
ax.loglog(n_cub,    t_cub,    "^-",  color="darkorange",  linewidth=2, markersize=6, label="CUB DeviceReduce::Sum")

ax.set_xlabel("n (number of elements)", fontsize=12)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("HW07 Task 1: Reduction — Thrust vs CUB vs HW05", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", linestyle="--", alpha=0.5)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved {OUT_PDF} and {OUT_PNG}")
