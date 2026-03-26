"""Plot HW09 Task 3: MPI ping-pong time vs message size n (log-log)."""

import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt("HW09/scaling_task3.dat")
n, ms = dat[:, 0], dat[:, 1]

# Bytes = n * 4 (float) for x-axis labelling
bytes_ = n * 4

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(n, ms, marker="o", linewidth=1.5, label="t0 + t1")
ax.set_xlabel("Message size n (number of floats)")
ax.set_ylabel("Total time (ms)")
ax.set_title("HW09 Task 3 — MPI ping-pong time vs. message size")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("HW09/task3.pdf")
print("Saved HW09/task3.pdf")
