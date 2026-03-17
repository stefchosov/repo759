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
│   ├── mmul.cu         # cuBLAS implementation
│   ├── scan.cu         # Hillis-Steele inclusive scan + host wrapper
│   ├── task1.cu        # Test harness: cuBLAS mmul
│   └── task2.cu        # Test harness: inclusive scan
├── sbatch/
│   ├── task1_hw6.sh           # Single run, task 1
│   ├── task1_hw6_scaling.sh   # Scaling study, task 1
│   ├── task2_hw6.sh           # Single run, task 2
│   ├── task2_hw6_scaling.sh   # Scaling study, task 2
│   └── task2_hw6_memcheck.sh  # compute-sanitizer run for task 2 (problem 2a)
└── Scripts/
    ├── plot_hw6_task1.py   # Plot task 1 scaling results
    ├── plot_hw6_task2.py   # Plot task 2 scaling results
    └── RUNBOOK.md          # This file
```

---

## Prerequisites

- CUDA toolkit available (tested with CUDA 12.x; `nvcc` must be on PATH)
- Python 3 with `numpy` and `matplotlib` for plotting
- On Euler: `module load nvidia/cuda/13.0` if nvcc is not already on PATH

---

## Task 1 — cuBLAS Matrix Multiplication

### What it does
`HW06/mmul.cu` wraps a single `cublasSgemm` call to compute **C := A·B + C**
for n×n column-major float matrices.  `task1.cu` is the timing harness.

### Output format
```
C[n*n - 1]   # last element of the result matrix
<time_ms>    # average time per mmul call in milliseconds
```

### Quick single-run test (interactive node)
```bash
cd /path/to/repo759

# Compile
nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o task1

# Run with n=1024, n_tests=10
./task1 1024 10
```

### Single-run via SLURM
```bash
sbatch sbatch/task1_hw6.sh           # default n=1024, n_tests=10
sbatch sbatch/task1_hw6.sh 4096 5    # custom n and n_tests
```

### Scaling study via SLURM
```bash
sbatch sbatch/task1_hw6_scaling.sh
# Produces: HW06/scaling_task1.dat (columns: n  avg_time_ms)
```

### Plot
```bash
python3 Scripts/plot_hw6_task1.py
# Saves: HW06/task1.pdf  HW06/task1.png
```

---

## Task 2 — Hillis-Steele Inclusive Scan

### What it does
`HW06/scan.cu` implements an inclusive prefix scan using the Hillis-Steele
algorithm in three GPU phases:
1. **Local scan** — each block independently scans its `threads_per_block`-element chunk.
2. **Block-sum scan** — a single block scans the per-block sums.
3. **Offset addition** — each block adds the previous block's cumulative sum.

Assumption: `n <= threads_per_block * threads_per_block`.

### Output format
```
output[n - 1]   # last element of the inclusive prefix sum
<time_ms>       # wall time of the scan() call in milliseconds
```

### Single-run via SLURM
```bash
sbatch sbatch/task2_hw6.sh              # default n=1024, tpb=1024
sbatch sbatch/task2_hw6.sh 65536 256   # custom n and tpb
```

### Scaling study via SLURM
```bash
sbatch sbatch/task2_hw6_scaling.sh
# Produces:
#   HW06/scaling_task2.dat (columns: n  time_ms, tpb fixed at 1024)
```

### cuda-memcheck / compute-sanitizer (Problem 2a)
```bash
sbatch sbatch/task2_hw6_memcheck.sh
# Output in hw6_task2_memcheck_<jobid>.out — include this in your Canvas submission
```

### Plot
```bash
python3 Scripts/plot_hw6_task2.py
# Saves: HW06/task2.pdf  HW06/task2.png
```

---

## Common issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `nvcc: command not found` | CUDA module not loaded | `module load nvidia/cuda/13.0` |
| `CUBLAS_STATUS_NOT_INITIALIZED` | cuBLAS handle not created | Ensure `cublasCreate` is called before `mmul` |
| `cudaErrorInvalidValue` in scan | n > tpb² | Reduce n or increase tpb |
| OOM on large n | GPU or host memory exhausted | Request `--mem=16G` in the sbatch header |

---

## Compilation flags reference

```bash
# Task 1 (cuBLAS)
nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o task1

# Task 2 (pure CUDA)
nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2
```

Both binaries must be compiled from the **HW06/** directory.

---

---

# HW07 Runbook

---

## Directory layout

```
repo759/
├── HW07/
│   ├── count.cuh         # Thrust count interface (provided)
│   ├── task1_thrust.cu   # Thrust reduce harness
│   ├── task1_cub.cu      # CUB reduce harness
│   ├── count.cu          # Thrust count implementation
│   ├── task2.cu          # Test harness for count
│   └── task3.cpp         # OpenMP factorial
├── sbatch/
│   ├── task1_thrust_hw7.sh
│   ├── task1_thrust_hw7_scaling.sh
│   ├── task1_cub_hw7.sh
│   ├── task1_cub_hw7_scaling.sh
│   ├── task2_hw7.sh
│   ├── task2_hw7_scaling.sh
│   └── task3_hw7.sh
└── Scripts/
    ├── plot_hw7_task1.py
    └── plot_hw7_task2.py
```

## Running

```bash
# Compile commands (from HW07/)
nvcc task1_thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_thrust
nvcc task1_cub.cu    -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_cub
nvcc task2.cu count.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2
g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# Scaling jobs
sbatch sbatch/task1_thrust_hw7_scaling.sh   # → HW07/scaling_task1_thrust.dat
sbatch sbatch/task1_cub_hw7_scaling.sh      # → HW07/scaling_task1_cub.dat
sbatch sbatch/task2_hw7_scaling.sh          # → HW07/scaling_task2.dat
sbatch sbatch/task3_hw7.sh

# Plots (from repo root after scp of .dat files)
python3 Scripts/plot_hw7_task1.py   # → HW07/task1.pdf
python3 Scripts/plot_hw7_task2.py   # → HW07/task2.pdf
```

---

---

# HW08 Runbook

OpenMP parallelism assignment — no GPU required.

---

## Directory layout

```
repo759/
├── HW08/
│   ├── matmul.h        # provided
│   ├── convolution.h   # provided
│   ├── msort.h         # provided
│   ├── matmul.cpp      # parallel mmul2 (OpenMP parallel for)
│   ├── convolution.cpp # parallel convolve (OpenMP collapse)
│   ├── msort.cpp       # parallel merge sort (OpenMP tasks)
│   ├── task1.cpp       # harness: mmul, args: n t
│   ├── task2.cpp       # harness: convolve, args: n t
│   └── task3.cpp       # harness: msort, args: n t ts
├── sbatch/
│   ├── task1_hw8.sh
│   ├── task1_hw8_scaling.sh
│   ├── task2_hw8.sh
│   ├── task2_hw8_scaling.sh
│   ├── task3_hw8_ts.sh
│   └── task3_hw8_t.sh
└── Scripts/
    ├── plot_hw8_task1.py
    ├── plot_hw8_task2.py
    ├── plot_hw8_task3_ts.py
    └── plot_hw8_task3_t.py
```

## Key differences from HW06/HW07

- **No GPU, no `module load`** — g++ with `-fopenmp` only
- **No CUDA events** — timing uses `std::chrono::high_resolution_clock`
- **`--cpus-per-task=20`, no `--gres=gpu:1`** in all sbatch scripts

## Step-by-step

### Step 1 — Run scaling studies on Euler

```bash
git pull

# Task 1: mmul, n=1024, t=1..20
sbatch sbatch/task1_hw8_scaling.sh        # → HW08/scaling_task1.dat

# Task 2: convolve, n=1024, t=1..20
sbatch sbatch/task2_hw8_scaling.sh        # → HW08/scaling_task2.dat

# Task 3a: msort threshold study, n=10^6, t=8, ts=2^1..2^10
sbatch sbatch/task3_hw8_ts.sh             # → HW08/scaling_task3_ts.dat
```

### Step 2 — Find best ts and run thread scaling

After `task3_hw8_ts.sh` completes:
```bash
# scp the ts data, then find the best ts:
python3 Scripts/plot_hw8_task3_ts.py      # prints "Best ts = <value>"

# Then run the thread scaling with that ts:
sbatch sbatch/task3_hw8_t.sh <best_ts>   # → HW08/scaling_task3_t.dat
```

### Step 3 — scp data files locally

```bash
scp sarsov@euler.engr.wisc.edu:repo759/HW08/*.dat .
mv scaling_task1.dat scaling_task2.dat scaling_task3_ts.dat scaling_task3_t.dat HW08/
```

### Step 4 — Generate plots

```bash
python3 Scripts/plot_hw8_task1.py      # → HW08/hw8_task1.pdf
python3 Scripts/plot_hw8_task2.py      # → HW08/hw8_task2.pdf
python3 Scripts/plot_hw8_task3_ts.py   # → HW08/hw8_task3_ts.pdf
python3 Scripts/plot_hw8_task3_t.py    # → HW08/hw8_task3_t.pdf
```

### Step 5 — Written answers (assignment8.txt)

The assignment asks for written discussion in Canvas for:
- **Task 2c**: discuss observations from the convolve plot — why does speedup
  plateau after a certain number of threads?
- **Task 3**: no explicit written question beyond generating the two plots.

Key points for Task 2c discussion:
- Speedup plateaus because the problem becomes memory-bandwidth bound before
  all 20 threads are saturated — a 1024×1024 convolution with a 3×3 mask
  has very little compute per memory access, so adding threads past the
  memory-bandwidth saturation point yields diminishing returns.
- Amdahl's Law also limits speedup if any sequential portions remain
  (e.g., memory allocation, filling arrays).

## Compilation reference (from HW08/)

```bash
g++ task1.cpp matmul.cpp     -Wall -O3 -std=c++17 -o task1 -fopenmp
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp
g++ task3.cpp msort.cpp      -Wall -O3 -std=c++17 -o task3 -fopenmp
```

## Common issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Speedup < 1 at t=2 | NUMA effects or small n | Try larger n |
| task3 crashes | n too large for stack | Already uses heap (`new int[n]`) — check memory limit |
| Wrong sort order | threshold=1 causes degenerate recursion | Use ts ≥ 2 |
