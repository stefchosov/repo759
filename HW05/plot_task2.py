#!/usr/bin/env python3
"""Generate scaling plot for HW05 Task 2 (parallel reduction)."""

import numpy as np
import matplotlib.pyplot as plt
import sys

dat_file = sys.argv[1] if len(sys.argv) > 1 else "scaling_task2.dat"

data = np.loadtxt(dat_file, comments="#")
N        = data[:, 0]
time_1024 = data[:, 1]
time_256  = data[:, 2]

fig, ax = plt.subplots(figsize=(9, 6))
ax.loglog(N, time_1024, "o-", label="threads_per_block = 1024", linewidth=2, markersize=6)
ax.loglog(N, time_256,  "s-", label="threads_per_block = 256",  linewidth=2, markersize=6)

ax.set_xlabel("Array size N", fontsize=12)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("Parallel Reduction — Time vs N", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.3)

out_pdf = dat_file.replace(".dat", ".pdf")
out_png = dat_file.replace(".dat", ".png")
plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Saved {out_pdf} and {out_png}")
