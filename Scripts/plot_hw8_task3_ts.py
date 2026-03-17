"""
Plot HW08 Task 3: msort time vs threshold ts (linear-log scale).
Reads:  HW08/scaling_task3_ts.dat  (columns: ts  time_ms)
Saves:  HW08/hw8_task3_ts.pdf  HW08/hw8_task3_ts.png

Usage (from repo root):
    python3 Scripts/plot_hw8_task3_ts.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

DATA    = os.path.join("HW08", "scaling_task3_ts.dat")
OUT_PDF = os.path.join("HW08", "hw8_task3_ts.pdf")
OUT_PNG = os.path.join("HW08", "hw8_task3_ts.png")

data    = np.loadtxt(DATA, comments="#")
ts_vals = data[:, 0].astype(int)
time_ms = data[:, 1]

best_idx = np.argmin(time_ms)
best_ts  = ts_vals[best_idx]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ts_vals, time_ms, "o-", color="steelblue", linewidth=2, markersize=6)
ax.axvline(best_ts, color="red", linestyle="--", linewidth=1.5,
           label=f"Best ts = {best_ts}")

ax.set_xscale("log", base=2)
ax.set_xlabel("Threshold ts", fontsize=12)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("HW08 Task 3: msort Time vs Threshold  (n=10⁶, t=8)", fontsize=12)
ax.set_xticks(ts_vals)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.legend(fontsize=10)
ax.grid(True, which="both", linestyle="--", alpha=0.5)

import matplotlib.ticker

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved {OUT_PDF} and {OUT_PNG}")
print(f"Best ts = {best_ts} ({time_ms[best_idx]:.3f} ms)")
