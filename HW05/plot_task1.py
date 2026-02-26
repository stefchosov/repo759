#!/usr/bin/env python3
"""Generate scaling plot for HW05 Task 1 (tiled matmul)."""

import numpy as np
import matplotlib.pyplot as plt
import sys

dat_file = sys.argv[1] if len(sys.argv) > 1 else "scaling_task1_16.dat"

data = np.loadtxt(dat_file, comments="#")
n          = data[:, 0]
time_int   = data[:, 1]
time_float = data[:, 2]
time_dbl   = data[:, 3]

fig, ax = plt.subplots(figsize=(9, 6))
ax.loglog(n, time_int,   "o-", label="int",    linewidth=2, markersize=6)
ax.loglog(n, time_float, "s-", label="float",  linewidth=2, markersize=6)
ax.loglog(n, time_dbl,   "^-", label="double", linewidth=2, markersize=6)

ax.set_xlabel("Matrix dimension n", fontsize=12)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("Tiled Matrix Multiplication — Time vs n", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.3)

out_pdf = dat_file.replace(".dat", ".pdf")
out_png = dat_file.replace(".dat", ".png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Saved {out_pdf} and {out_png}")
