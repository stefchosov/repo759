"""Plot HW09 Task 1 scaling: cluster time vs thread count (linear-linear)."""

import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt("HW09/scaling_task1.dat")
t, ms = dat[:, 0], dat[:, 1]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(t, ms, marker="o", linewidth=1.5, label="cluster time")
ax.set_xlabel("Number of threads (t)")
ax.set_ylabel("Time (ms)")
ax.set_title("HW09 Task 1 — cluster() time vs. threads (n=5 040 000)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("HW09/task1.pdf")
print("Saved HW09/task1.pdf")
