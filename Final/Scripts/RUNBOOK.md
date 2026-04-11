# Final Project Runbook

**Title:** CPU vs GPU Performance and Truncated-Collision Analysis of MD5, SHA-1, and SHA-256
**Student:** Stefan Arsov — ME759 Spring 2026

---

## Directory layout

```
Final/
├── Code/
│   ├── md5.cuh / md5.cu          # CUDA MD5 kernel + host wrapper
│   ├── sha1.cuh / sha1.cu        # CUDA SHA-1 kernel + host wrapper
│   ├── sha256.cuh / sha256.cu    # CUDA SHA-256 kernel + host wrapper
│   ├── md5_cpu.h / md5_cpu.cpp   # CPU MD5 implementation
│   ├── sha1_cpu.h / sha1_cpu.cpp
│   ├── sha256_cpu.h / sha256_cpu.cpp
│   ├── task1.cu   # CPU vs GPU throughput benchmark (args: n algo)
│   ├── task2.cu   # GPU thread-scaling benchmark (args: n tpb algo)
│   └── task3.cu   # Truncated collision search (args: bits algo)
├── sbatch/
│   ├── task1_final.sh             # single-run: throughput
│   ├── task1_final_scaling.sh     # scaling: vary n, all three algos
│   ├── task2_final.sh             # single-run: thread scaling
│   ├── task2_final_scaling.sh     # scaling: vary tpb
│   └── task3_final.sh             # collision search, all truncation sizes
├── Scripts/
│   ├── plot_final_task1.py        # throughput bar/line charts
│   ├── plot_final_task2.py        # thread-scaling speedup curves
│   ├── plot_final_task3.py        # collision time vs truncation bits
│   └── RUNBOOK.md                 # this file
├── Data/                          # .dat output files from sbatch jobs
└── Report/                        # final report PDF / source
```

---

## CPU Testing Walkthrough (Task 1 — serial + OpenMP)

End-to-end steps to compile, verify correctness, and run the scaling study
for the CPU-only implementation on Euler before any GPU work.

---

### Step 1 — Get the latest code on Euler

```bash
ssh sarsov@euler.engr.wisc.edu
cd ~/repo759
git pull
```

---

### Step 2 — Compile

Run from the repo root (no module loads needed — g++ with OpenMP is available by default):

```bash
g++ -O3 -std=c++17 -fopenmp \
    -o Final/task1 \
    Final/Code/task1.cpp \
    Final/Code/md5_cpu.cpp \
    Final/Code/sha1_cpu.cpp \
    Final/Code/sha256_cpu.cpp
```

Expected: no output, binary at `Final/task1`.

---

### Step 3 — Verify correctness against reference values

The check bytes (lines 1, 4, 7 of output) are the last byte of each algorithm's
hash of the 8-byte all-zeros input.  Compute the expected values with Python:

```bash
python3 -c "
import hashlib
data = bytes(8)
print('Expected MD5   check byte:', hashlib.md5(data).digest()[-1])
print('Expected SHA1  check byte:', hashlib.sha1(data).digest()[-1])
print('Expected SHA256 check byte:', hashlib.sha256(data).digest()[-1])
"
```

Then run task1 with n=1 (single hash) and compare:

```bash
./Final/task1 1 1
```

Lines 1, 4, 7 of the output must match the Python-printed values exactly.
If they do not, the padding or byte-order in the corresponding implementation
is wrong — do not proceed to the scaling study until this passes.

---

### Step 4 — Quick sanity run (interactive)

```bash
./Final/task1 1000000 20
```

Expected output (9 lines):
```
<md5_check_byte>       ← integer 0–255
<md5_serial_ms>        ← ~10–50 ms for 10^6 hashes
<md5_omp_ms>           ← should be ~10–20x faster than serial
<sha1_check_byte>
<sha1_serial_ms>       ← slightly slower than MD5
<sha1_omp_ms>
<sha256_check_byte>
<sha256_serial_ms>     ← slowest of the three
<sha256_omp_ms>
```

Sanity checks:
- OMP time < serial time for all three algos (otherwise OpenMP isn't engaging)
- SHA-256 serial > SHA-1 serial > MD5 serial  (complexity order)
- All check bytes are non-zero integers (a zero could indicate a bug, though it is theoretically valid)

---

### Step 5 — Submit the scaling study

```bash
sbatch Final/sbatch/task1_final_scaling.sh
```

This sweeps n = 100 K → 100 M with t=20 threads and writes:
```
Final/Data/scaling_task1_cpu.dat
```

Monitor progress:
```bash
squeue -u sarsov          # check job status
# Once RUNNING, tail the live output:
tail -f Final/sbatch/logs/task1_final_scaling_<jobid>.out
```

Typical wall time: 5–10 minutes.

---

### Step 6 — Retrieve data

From your local machine:

```bash
scp sarsov@euler.engr.wisc.edu:repo759/Final/Data/scaling_task1_cpu.dat Final/Data/
```

---

### Step 7 — Generate the plot

```bash
python3 Final/Scripts/plot_final_task1.py
```

Produces:
- `Final/Data/task1_throughput.pdf`
- `Final/Data/task1_throughput.png`

The GPU bars will be absent until `scaling_task1_gpu.dat` exists — that is expected.

---

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Compile error: `fopenmp not recognized` | Old g++ | `module load gcc/12` or similar |
| Check byte mismatch for MD5 | Endianness bug in `md5_block` output | Verify little-endian word write in `md5_cpu.cpp:87` |
| Check byte mismatch for SHA-1/SHA-256 | Endianness bug | Verify big-endian word write in the respective `_cpu.cpp` |
| OMP time ≥ serial time | Thread count not taking effect | Confirm `--cpus-per-task=20` in sbatch; check `OMP_NUM_THREADS` |
| `.dat` file not created | Job failed before writing | Check `task1_final_scaling_<jobid>.out` for error |
| `plot_final_task1.py` crashes | Missing `scaling_task1_cpu.dat` | Complete Step 5–6 first |

---

## Task 1 — Serial vs OpenMP vs GPU Throughput

Measures hashes/second for MD5, SHA-1, and SHA-256 across three execution modes:
serial CPU, OpenMP CPU (t threads), and CUDA GPU (added later in task1.cu).

### Output format (task1.cpp — CPU only)
```
<md5_check_byte>      # last byte of MD5(0) — correctness check
<md5_serial_ms>       # wall time for n serial MD5 hashes
<md5_omp_ms>          # wall time for n OpenMP MD5 hashes
<sha1_check_byte>
<sha1_serial_ms>
<sha1_omp_ms>
<sha256_check_byte>
<sha256_serial_ms>
<sha256_omp_ms>
```

### Compile (from repo root)
```bash
g++ -O3 -std=c++17 -fopenmp \
    -o Final/task1 \
    Final/Code/task1.cpp \
    Final/Code/md5_cpu.cpp \
    Final/Code/sha1_cpu.cpp \
    Final/Code/sha256_cpu.cpp
```

### Scaling job
```bash
sbatch Final/sbatch/task1_final_scaling.sh   # → Final/Data/scaling_task1_md5.dat etc.
```

### Plot
```bash
python3 Final/Scripts/plot_final_task1.py    # → Final/Data/task1.pdf
```

---

## Task 2 — GPU Thread-Count Scaling

Measures GPU throughput as threads_per_block varies, for each algorithm.

### Compile
```bash
nvcc -O3 -std=c++17 -o Final/task2 Final/Code/task2.cu Final/Code/md5.cu Final/Code/sha1.cu Final/Code/sha256.cu
```

### Scaling job
```bash
sbatch Final/sbatch/task2_final_scaling.sh   # → Final/Data/scaling_task2_<algo>.dat
```

### Plot
```bash
python3 Final/Scripts/plot_final_task2.py
```

---

## Task 3 — Truncated Collision Search (Birthday Paradox)

Searches for collisions in hash outputs truncated to 16, 20, 24, 28, 32 bits.
Expected collision counts follow the birthday paradox: ~2^(bits/2) samples.

| Truncation | Expected samples |
|-----------|-----------------|
| 16 bits   | ~256            |
| 20 bits   | ~1,024          |
| 24 bits   | ~4,096          |
| 28 bits   | ~16,384         |
| 32 bits   | ~65,536         |

### Compile
```bash
nvcc -O3 -std=c++17 -o Final/task3 Final/Code/task3.cu Final/Code/md5.cu Final/Code/sha1.cu Final/Code/sha256.cu
```

### Run
```bash
sbatch Final/sbatch/task3_final.sh   # → Final/Data/scaling_task3_<algo>.dat
```

### Plot
```bash
python3 Final/Scripts/plot_final_task3.py   # → Final/Data/task3.pdf
```

---

## Data files naming convention

```
Final/Data/scaling_task1_md5.dat       # columns: n  cpu_ms  gpu_ms  speedup
Final/Data/scaling_task1_sha1.dat
Final/Data/scaling_task1_sha256.dat
Final/Data/scaling_task2_md5.dat       # columns: tpb  throughput_GHps
Final/Data/scaling_task2_sha1.dat
Final/Data/scaling_task2_sha256.dat
Final/Data/scaling_task3_md5.dat       # columns: bits  samples  time_ms
Final/Data/scaling_task3_sha1.dat
Final/Data/scaling_task3_sha256.dat
```

---

## Euler cluster notes

- Partition: `instruction`
- GPU: `--gres=gpu:1`
- Memory: `--mem=16G`
- CUDA module: `module load nvidia/cuda/12.0` (or latest available)
- Compile from repo root: `nvcc -O3 -std=c++17 ...`
