#!/usr/bin/env python3
"""
Generate task3.pdf plot showing vscale kernel performance
with 512 vs 16 threads per block
"""

import numpy as np
import matplotlib.pyplot as plt

# Read data files
data_512 = np.loadtxt('../HW03/scaling_512.dat', skiprows=1)
data_16 = np.loadtxt('../HW03/scaling_16.dat', skiprows=1)

# Extract n and time values
n_512 = data_512[:, 0]
time_512 = data_512[:, 1]

n_16 = data_16[:, 0]
time_16 = data_16[:, 1]

# Create plot
plt.figure(figsize=(10, 6))
plt.loglog(n_512, time_512, 'o-', label='512 threads/block', linewidth=2, markersize=6)
plt.loglog(n_16, time_16, 's-', label='16 threads/block', linewidth=2, markersize=6)

plt.xlabel('Array Size (n)', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.title('vscale Kernel Performance: 512 vs 16 Threads per Block', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=11)

# Save to HW03 directory
plt.savefig('../HW03/task3.pdf', bbox_inches='tight', dpi=300)
print("Plot saved to HW03/task3.pdf")

# Also save as PNG for easy viewing
plt.savefig('../HW03/task3.png', bbox_inches='tight', dpi=300)
print("Plot saved to HW03/task3.png")

plt.show()
