#!/usr/bin/env python3
"""
Plotting script for HW04 Task 2 (1D Stencil) scaling study
Reads scaling data and generates log-log plots
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# File paths
data_dir = "../HW04"
output_dir = "../HW04"

# Read data files
data_1024 = np.loadtxt(os.path.join(data_dir, "scaling_task2_1024.dat"), comments='#')
data_512 = np.loadtxt(os.path.join(data_dir, "scaling_task2_512.dat"), comments='#')

# Extract n and time values
n_1024 = data_1024[:, 0]
time_1024 = data_1024[:, 1]

n_512 = data_512[:, 0]
time_512 = data_512[:, 1]

# Create log-log plot
plt.figure(figsize=(10, 7))

plt.loglog(n_1024, time_1024, 'o-', label='1024 threads/block', linewidth=2, markersize=8)
plt.loglog(n_512, time_512, 's-', label='512 threads/block', linewidth=2, markersize=8)

plt.xlabel('Array size (n)', fontsize=14)
plt.ylabel('Execution time (ms)', fontsize=14)
plt.title('Task 2: 1D Stencil Performance Scaling (R=128)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', alpha=0.3)

# Save plots
pdf_path = os.path.join(output_dir, "task2.pdf")
png_path = os.path.join(output_dir, "task2.png")

plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
plt.savefig(png_path, bbox_inches='tight', dpi=300)

print(f"Plots saved to:")
print(f"  {pdf_path}")
print(f"  {png_path}")

plt.close()
