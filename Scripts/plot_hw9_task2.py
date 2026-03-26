"""Plot HW09 Task 2: montecarlo time vs threads, with and without simd."""

import numpy as np
import matplotlib.pyplot as plt

simd   = np.loadtxt("HW09/scaling_task2_simd.dat")
nosimd = np.loadtxt("HW09/scaling_task2_nosimd.dat")

t_s, ms_s = simd[:, 0],   simd[:, 1]
t_n, ms_n = nosimd[:, 0], nosimd[:, 1]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(t_s, ms_s, marker="o", linewidth=1.5, label="with simd")
ax.plot(t_n, ms_n, marker="s", linewidth=1.5, label="without simd", linestyle="--")
ax.set_xlabel("Number of threads (t)")
ax.set_ylabel("Time (ms)")
ax.set_title("HW09 Task 2 — montecarlo() time vs. threads (n=10⁶)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("HW09/task2.pdf")
print("Saved HW09/task2.pdf")
