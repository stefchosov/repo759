"""
Plot HW06 Task 1 scaling results: cuBLAS matrix multiplication time vs n.
Reads HW06/scaling_task1.dat and saves the figure to HW06/task1.pdf + task1.png.

Usage (after running the scaling sbatch job):
    python3 Scripts/plot_hw6_task1.py
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster use
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_FILE = os.path.join("HW06", "scaling_task1.dat")
OUT_PDF   = os.path.join("HW06", "task1.pdf")
OUT_PNG   = os.path.join("HW06", "task1.png")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = np.loadtxt(DATA_FILE, comments="#")
n_vals   = data[:, 0].astype(int)
time_ms  = data[:, 1]

# Compute peak float32 FLOPs for each n: 2*n^3 operations
flops    = 2.0 * n_vals.astype(float) ** 3
gflops   = flops / (time_ms * 1e-3) / 1e9   # GFLOP/s

# ---------------------------------------------------------------------------
# Plot 1: time vs n (log-log)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.loglog(n_vals, time_ms, "o-", color="steelblue", linewidth=2, markersize=6,
          label="cuBLAS SGEMM")
ax.set_xlabel("Matrix dimension n", fontsize=12)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("HW06 Task 1: cuBLAS SGEMM Scaling", fontsize=12)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(fontsize=10)

# Annotate ideal O(n^3) slope for reference
n_ref = np.array([n_vals[0], n_vals[-1]], dtype=float)
scale = time_ms[0] / (n_vals[0] ** 3)
ax.loglog(n_ref, scale * n_ref ** 3, "k--", linewidth=1, label=r"$O(n^3)$")
ax.legend(fontsize=10)

# ---------------------------------------------------------------------------
# Plot 2: achieved GFLOP/s vs n
# ---------------------------------------------------------------------------
ax2 = axes[1]
ax2.semilogx(n_vals, gflops, "s-", color="darkorange", linewidth=2, markersize=6)
ax2.set_xlabel("Matrix dimension n", fontsize=12)
ax2.set_ylabel("Performance (GFLOP/s)", fontsize=12)
ax2.set_title("HW06 Task 1: Achieved Throughput", fontsize=12)
ax2.grid(True, which="both", linestyle="--", alpha=0.5)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved {OUT_PDF} and {OUT_PNG}")
