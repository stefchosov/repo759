# HW06 Runbook

End-to-end instructions for compiling, running, and plotting the HW06 assignment
on the UW-Madison Euler cluster (or any SLURM + CUDA cluster).

---

## Directory layout

```
repo759/
├── HW06/
│   ├── mmul.h          # cuBLAS mmul interface (provided)
│   ├── scan.cuh        # Hillis-Steele scan interface (provided)
│   ├── mmul.cpp        # cuBLAS implementation
│   ├── scan.cu         # Hillis-Steele inclusive scan + host wrapper
│   ├── task1.cu        # Test harness: cuBLAS mmul
│   └── task2.cu        # Test harness: inclusive scan
├── sbatch/
│   ├── task1_hw6.sh           # Single run, task 1
│   ├── task1_hw6_scaling.sh   # Scaling study, task 1
│   ├── task2_hw6.sh           # Single run, task 2
│   └── task2_hw6_scaling.sh   # Scaling study, task 2
└── Scripts/
    ├── plot_hw6_task1.py   # Plot task 1 scaling results
    ├── plot_hw6_task2.py   # Plot task 2 scaling results
    └── RUNBOOK.md          # This file
```

---

## Prerequisites

- CUDA toolkit available (tested with CUDA 12.x; `nvcc` must be on PATH)
- Python 3 with `numpy` and `matplotlib` for plotting
- On Euler: `module load cuda` if nvcc is not already on PATH

---

## Task 1 — cuBLAS Matrix Multiplication

### What it does
`HW06/mmul.cpp` wraps a single `cublasSgemm` call to compute **C := A·B + C**
for n×n column-major float matrices.  `task1.cu` is the timing harness.

### Output format
```
C[n*n - 1]   # last element of the result matrix
<time_ms>    # wall time of the mmul() call in milliseconds
```

### Quick single-run test (interactive node)
```bash
cd /path/to/repo759

# Compile
nvcc -O3 -std=c++14 -o HW06/task1 HW06/task1.cu HW06/mmul.cpp -lcublas

# Run with n=1024
./HW06/task1 1024
```

### Single-run via SLURM
```bash
# Default n=1024
sbatch sbatch/task1_hw6.sh

# Custom n=4096
sbatch sbatch/task1_hw6.sh 4096
```

### Scaling study via SLURM
```bash
sbatch sbatch/task1_hw6_scaling.sh
# Produces: HW06/scaling_task1.dat (columns: n  time_ms)
```

### Plot
Run from the repo root **after** the scaling job completes:
```bash
python3 Scripts/plot_hw6_task1.py
# Saves: HW06/task1.pdf  HW06/task1.png
```

---

## Task 2 — Hillis-Steele Inclusive Scan

### What it does
`HW06/scan.cu` implements an inclusive prefix scan using the Hillis-Steele
algorithm.  The computation is split into three GPU phases:

1. **Local scan** — each block independently scans its `threads_per_block`-element chunk.
2. **Block-sum scan** — a single block scans the per-block sums (≤ `threads_per_block` values, always fits in one block given n ≤ tpb²).
3. **Offset addition** — each block (except block 0) adds the previous block's cumulative sum.

Assumption: `n <= threads_per_block * threads_per_block`.

### Output format
```
output[n - 1]   # last element of the inclusive prefix sum
<time_ms>       # wall time of the scan() call in milliseconds
```

### Quick single-run test (interactive node)
```bash
# Compile
nvcc -O3 -std=c++14 -o HW06/task2 HW06/task2.cu HW06/scan.cu

# Run: n=1048576, tpb=1024  (n == tpb^2, largest valid input)
./HW06/task2 1048576 1024
```

### Correctness sanity check (small n)
```bash
nvcc -O3 -std=c++14 -o HW06/task2 HW06/task2.cu HW06/scan.cu
./HW06/task2 8 4   # n=8, tpb=4  (2 blocks)
# output[7] should equal the sum of all 8 random inputs.
# Verify independently: generate the same 8 values with seed 759 and sum them.
```

### Single-run via SLURM
```bash
# Default: n=1048576, tpb=1024
sbatch sbatch/task2_hw6.sh

# Custom: n=65536, tpb=256
sbatch sbatch/task2_hw6.sh 65536 256
```

### Scaling study via SLURM
```bash
sbatch sbatch/task2_hw6_scaling.sh
# Produces:
#   HW06/scaling_task2_n.dat    (columns: n  time_ms, tpb fixed at 1024)
#   HW06/scaling_task2_tpb.dat  (columns: tpb  time_ms, n fixed at 1048576)
```

### Plot
Run from the repo root **after** the scaling job completes:
```bash
python3 Scripts/plot_hw6_task2.py
# Saves: HW06/task2.pdf  HW06/task2.png
```

---

## Common issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `nvcc: command not found` | CUDA module not loaded | `module load cuda` |
| `CUBLAS_STATUS_NOT_INITIALIZED` | cuBLAS handle not created | Ensure `cublasCreate` is called before `mmul` |
| `cudaErrorInvalidValue` in scan | n > tpb² | Reduce n or increase tpb |
| Wrong prefix-sum result | Compiler optimised away `__syncthreads` | Ensure `-O3` (nvcc respects `__syncthreads` regardless) |
| OOM on large n | GPU or host memory exhausted | Request `--mem=16G` in the sbatch header |

---

## Compilation flags reference

```bash
# Task 1 (cuBLAS — links against -lcublas)
nvcc -O3 -std=c++14 -o HW06/task1 HW06/task1.cu HW06/mmul.cpp -lcublas

# Task 2 (pure CUDA — no extra libraries)
nvcc -O3 -std=c++14 -o HW06/task2 HW06/task2.cu HW06/scan.cu
```

Both binaries must be compiled from the **repo root** so that relative include
paths (`#include "mmul.h"`, `#include "scan.cuh"`) resolve correctly.
