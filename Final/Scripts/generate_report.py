"""
Generates Final/Report/FinalProject_Arsov.docx from template structure.
Run from repo root: python3 Final/Scripts/generate_report.py
"""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

doc = Document()

# ── Styles ────────────────────────────────────────────────────────────────────
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)

def heading(text, level=1):
    p = doc.add_heading(text, level=level)
    p.style.font.size = Pt(12 if level > 1 else 14)
    return p

def body(text):
    p = doc.add_paragraph(text)
    p.style = doc.styles['Normal']
    return p

def bullet(text):
    p = doc.add_paragraph(text, style='List Bullet')
    return p

def add_table(headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True
    for ri, row in enumerate(rows):
        cells = t.rows[ri + 1].cells
        for ci, val in enumerate(row):
            cells[ci].text = str(val)
    doc.add_paragraph()

# ── Title page ────────────────────────────────────────────────────────────────
title = doc.add_heading('Spring 2026 ME/CS/ECE 759 Final Project Report', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph('University of Wisconsin–Madison')
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

proj = doc.add_paragraph('CPU vs GPU Performance and Truncated-Collision Analysis\nof MD5, SHA-1, and SHA-256')
proj.alignment = WD_ALIGN_PARAGRAPH.CENTER
proj.runs[0].bold = True

author = doc.add_paragraph('Stefan Arsov\nMay 5, 2026')
author.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_page_break()

# ── Abstract ──────────────────────────────────────────────────────────────────
heading('Abstract')
body(
    "This project benchmarks MD5, SHA-1, and SHA-256 cryptographic hash functions "
    "across three execution models — CPU serial, CPU parallel (OpenMP), and GPU (CUDA) "
    "— and investigates truncated-collision search as a vehicle for demonstrating "
    "GPU algorithmic advantages at scale. "
    "Task 1 measures raw hash throughput as a function of batch size n, establishing "
    "GPU vs CPU speedup baselines (up to 440× for SHA-1 at n=10M). "
    "Task 2 sweeps threads-per-block (tpb) from 32 to 1024, revealing that register "
    "pressure limits occupancy and makes tpb=32 optimal for all three algorithms. "
    "Task 3 implements truncated collision search over bit widths {16,24,32,40,48,56,64,72} "
    "using three approaches: a sequential CPU unordered_map scan, a two-phase OpenMP "
    "parallel scan (12 threads, single-socket), and GPU Pollard's rho with distinguished "
    "points using 65,536 CUDA threads and O(1) memory per thread. The GPU is the only "
    "implementation capable of running at bits≥64; at bits=56 it achieves 260× speedup "
    "over CPU serial for MD5. A 128-bit arithmetic extension enables the GPU to reach "
    "bits=72, a range completely infeasible for any CPU approach. The OpenMP sweet spot "
    "at 12 threads is explained by NUMA topology, Amdahl's serial merge bottleneck, and "
    "L3 cache pressure. All experiments ran on the UW–Madison Euler cluster (NVIDIA GPU, "
    "CUDA 13.0, dual-socket CPU node)."
)
body("Link to Final Project git repo: https://github.com/stefchosov/repo759")

doc.add_page_break()

# ── General information ───────────────────────────────────────────────────────
heading('General Information')
bullet("Name: Stefan Arsov")
bullet("Email: sarsov@epic.com")
bullet("Home department: Computer Sciences")
bullet("Status: MS Student")
bullet("Teammate: None")
bullet("I release the ME759 Final Project code as open source and under a BSD3 license "
       "for unfettered use of it by any interested party.")

# ── 1. Problem Statement ──────────────────────────────────────────────────────
heading('1. Problem Statement')
body(
    "Cryptographic hash functions — MD5, SHA-1, and SHA-256 — are the computational "
    "backbone of password hashing, digital signatures, file integrity verification, "
    "and certificate chains. Their security rests on two properties: preimage resistance "
    "(hard to find input given output) and collision resistance (hard to find two inputs "
    "with the same output). While the full-length versions of these algorithms are "
    "cryptographically strong, truncated variants appear in many practical systems: "
    "SSH host key fingerprints use truncated MD5 or SHA-256; Git object IDs use "
    "truncated SHA-1; many authentication protocols use only the first N bits of a "
    "digest to reduce bandwidth."
)
body(
    "This project asks two questions: (1) How does GPU parallelism affect raw hash "
    "throughput relative to serial and OpenMP CPU implementations, and how does this "
    "vary by algorithm complexity and configuration? (2) How does GPU parallelism "
    "change the feasibility of truncated collision search — finding two distinct inputs "
    "whose N-bit truncated hashes collide — and at what bit width does the GPU's "
    "algorithmic advantage become decisive? The motivation is to understand the "
    "practical boundary between CPU-feasible and GPU-only cryptanalysis tasks, directly "
    "relevant to password cracking, rainbow table construction, and certificate spoofing "
    "scenarios."
)

# ── 2. Solution Description ───────────────────────────────────────────────────
heading('2. Solution Description')

heading('2.1 Code Structure', level=2)
body("All code lives under Final/Code/ in the repository:")
bullet("task1.cu — GPU throughput benchmark: launches n parallel hash kernels, "
       "measures time via cudaEvent_t, reports MH/s for each algorithm and tpb=256.")
bullet("task2.cu — tpb sweep: same workload as Task 1 but varies threads-per-block "
       "from 32 to 1024, reporting throughput at each configuration.")
bullet("task3.cu — collision search: CPU serial, OpenMP parallel, and GPU Pollard's "
       "rho implementations for bits in {16,24,32,40,48,56,64,72}.")
bullet("task4.cu — GPU Thrust-sort collision search: hash → thrust::sort_by_key → "
       "adjacent-duplicate kernel, bits in {16,24,32,40,48} (VRAM-capped).")
bullet("gpu_hashes_narrow.cuh — shared device hash functions used by task4.cu (and "
       "available for future GPU work).")
bullet("md5_cpu.cpp / sha1_cpu.cpp / sha256_cpu.cpp — portable CPU reference "
       "implementations used by both CPU and OpenMP paths in Task 3.")
bullet("md5_cpu.h / sha1_cpu.h / sha256_cpu.h — headers for the above.")

heading('2.2 Task 1 — Hash Throughput', level=2)
body(
    "Task 1 launches n CUDA threads, each computing one hash of an 8-byte "
    "little-endian encoding of its thread index. Three template kernels "
    "(template<int ALGO>) specialize at compile time for MD5, SHA-1, and SHA-256, "
    "eliminating runtime branches. The GPU hash functions operate entirely in "
    "registers — no shared memory, no global memory reads after the index load. "
    "Timing uses cudaEvent_t around the kernel call with cudaDeviceSynchronize() "
    "inside the called function. The experiment sweeps n from 100K to 10M to show "
    "the GPU break-even point relative to serial CPU."
)

heading('2.3 Task 2 — Threads-per-Block Sweep', level=2)
body(
    "Task 2 holds n=10,000,000 fixed and sweeps tpb in {32, 64, 128, 256, 512, 1024}. "
    "The same template kernel from Task 1 is reused with the tpb as a launch parameter. "
    "The key insight is that each hash kernel is register-heavy (SHA-256 maintains 9 "
    "32-bit working variables plus a 64-entry message schedule W[64]), so increasing "
    "tpb reduces the number of blocks the SM can schedule concurrently, reducing "
    "occupancy and hiding less latency."
)

heading('2.4 Task 3 — Truncated Collision Search', level=2)
body(
    "Task 3 implements three independent collision search strategies for truncated "
    "N-bit hashes, where a collision is two distinct inputs i ≠ j such that "
    "truncate(hash(i), N) == truncate(hash(j), N)."
)

body("CPU Serial: Sequential scan of inputs 0,1,2,... inserting each truncated hash "
     "into an std::unordered_map<uint64_t,uint64_t>. The first duplicate key is a "
     "collision. Limited to bits≤56 due to memory (bits=56 requires ~17 GB for the map, "
     "accommodated by --mem=96G in SLURM).")

body("OpenMP Parallel (12 threads): Two-phase approach per batch. Phase 1 (parallel): "
     "threads hash disjoint index ranges into pre-allocated hbuf/ibuf arrays with no "
     "synchronization. Phase 2 (serial): merge into the global map, stopping at first "
     "duplicate. Thread count is fixed at 12 — a dedicated OMP scaling experiment "
     "(sweeping 1–32 threads at bits={40,48}) showed peak performance at 12 due to: "
     "(1) NUMA boundary: Euler nodes are dual-socket; threads 13+ cross to the remote "
     "socket paying 2× DRAM latency; (2) Amdahl's serial merge: Phase 2 is fully serial, "
     "capping parallel speedup; (3) L3 cache pressure: the batch working set (~32 MB at "
     "bits=48) saturates per-socket L3 beyond 12 threads.")

body("GPU — Pollard's Rho with Distinguished Points: 65,536 CUDA threads each walk an "
     "independent Markov chain x_{n+1} = truncate(hash(x_n), N bits). A 'distinguished "
     "point' (DP) is any chain value x with its low N/4 bits zero. Two chains reaching "
     "the same DP from different starts imply a collision. This uses O(1) memory per "
     "thread — only an 8 MB DP buffer — and scales to bits=64 without hitting device "
     "memory limits. Template kernels eliminate warp divergence. For bits>64 a 128-bit "
     "extension was implemented: a uint128 struct (lo/hi pair), wide hash functions "
     "accepting 16-byte inputs, and a pollard_kernel_wide template, enabling bits=72.")

# ── 3. Results ────────────────────────────────────────────────────────────────
heading('3. Overview of Results')

heading('3.1 Task 1 — Throughput at n=10,000,000', level=2)
add_table(
    ['Algorithm', 'CPU Serial (MH/s)', 'OpenMP-4 (MH/s)', 'GPU (MH/s)', 'GPU/Serial Speedup'],
    [
        ['MD5',     '6.53',  '22.73', '2694.6', '413×'],
        ['SHA-1',   '2.78',   '9.34', '1224.8', '440×'],
        ['SHA-256', '3.94',   '9.54',  '658.9', '167×'],
    ]
)
body(
    "GPU throughput is 167–440× faster than serial CPU. OpenMP-4 achieves near-linear "
    "speedup (3.5×) for MD5 and SHA-1, and 2.4× for SHA-256 (limited by higher register "
    "pressure). At n<~200,000 the GPU is slower than serial due to kernel launch "
    "overhead (~2 ms); above this threshold GPU parallelism dominates. Plots are in "
    "Final/Data/task1/task1_throughput.pdf."
)

heading('3.2 Task 2 — Threads-per-Block', level=2)
add_table(
    ['tpb', 'MD5 (MH/s)', 'SHA-1 (MH/s)', 'SHA-256 (MH/s)'],
    [
        ['32',   '3021.1', '1201.9', '574.7'],
        ['64',   '2840.9', '1052.6', '559.3'],
        ['128',  '2717.4', '1066.1', '557.7'],
        ['256',  '2277.9', '1021.5', '537.3'],
        ['512',  '2197.8', '1003.0', '540.8'],
        ['1024', '2375.3', '1003.0', '557.4'],
    ]
)
body(
    "tpb=32 (one warp) is optimal for all three algorithms. Throughput decreases as "
    "tpb increases because larger blocks consume more of the SM register file, reducing "
    "concurrent warp occupancy. MD5 drops 27% from tpb=32 to tpb=512 (largest impact "
    "since it starts with highest occupancy); SHA-256 drops only 6% (already register-"
    "starved at tpb=32). Using tpb=32 vs the default tpb=256 gives 7–33% free speedup. "
    "Plot: Final/Data/task2/task2_tpb.pdf."
)

heading('3.3 Task 3 — Collision Search Timing', level=2)
body("Selected results (time in ms). CPU/OMP = -1 for bits≥64 (infeasible).")

add_table(
    ['Algo', 'Bits', 'CPU Serial (ms)', 'OMP-12 (ms)', 'GPU (ms)', 'GPU/CPU Speedup'],
    [
        ['MD5',     '32', '9.5',        '2.6',       '36.7',    '0.26×'],
        ['MD5',     '48', '4,234',      '2,327',     '544',     '7.8×'],
        ['MD5',     '56', '554,672',    '359,864',   '2,133',   '260×'],
        ['MD5',     '64', '—',          '—',         '8,488',   '—'],
        ['SHA-1',   '56', '74,417',     '35,765',    '2,375',   '31×'],
        ['SHA-256', '56', '316,526',    '181,146',   '1,930',   '164×'],
        ['SHA-256', '64', '—',          '—',         '7,585',   '—'],
    ]
)
body(
    "GPU speedup over CPU serial grows dramatically with bit width, from sub-1× at "
    "bits=32 (GPU overhead dominates at small search spaces) to 31–260× at bits=56. "
    "At bits=64 and bits=72 only the GPU runs — CPU memory requirements (~100 GB for "
    "bits=64 with unordered_map) make CPU collision search infeasible on available "
    "hardware. The OMP implementation achieves only 1.5–2.1× speedup over serial at "
    "bits=56, limited by the serial merge phase (Amdahl's Law) and NUMA penalties. "
    "The birthday paradox is validated by CPU data: cpu_count/expected is consistently "
    "within 1–2× of 1.0 for all algorithms and bit widths. Plots: "
    "Final/Data/task3/task3_time.pdf, task3_speedup.pdf, task3_birthday.pdf."
)

body(
    "The OMP thread-count scaling experiment (Final/Data/task3/omp_scaling.dat) "
    "confirmed the 12-thread sweet spot: speedup peaks at 12 and degrades beyond "
    "that due to NUMA penalties on the dual-socket Euler node. "
    "Plot: Final/Data/task3/omp_scaling_speedup.pdf."
)

heading('3.4 Task 4 — GPU Algorithm Comparison: Thrust Sort vs Pollard\'s rho', level=2)
body(
    "Task 4 implements an alternative GPU collision search using thrust::sort_by_key "
    "to compare against Task 3's Pollard's rho. Three-phase: hash a batch of inputs "
    "into device arrays, sort by hash key, scan for first adjacent equal-hash pair. "
    "Fully GPU-resident with no CPU-side merge phase. Two memory modes are tested:"
)
bullet(
    "Device mode (cudaMalloc): pure VRAM. Capped at bits=48 because the sort working "
    "set (4 × expected × 16 bytes) plus Thrust scratch saturates 8 GB GPU memory at "
    "bits ≥ 52."
)
bullet(
    "Unified memory mode (cudaMallocManaged): the buffers are visible to both GPU "
    "and host, with pages migrating across PCIe on demand. Lets us push the sweep "
    "to bits=52 and bits=56 — the working set spills into host RAM (--mem=96G) "
    "rather than OOMing — at the cost of PCIe-bound performance during sort."
)
body(
    "Adaptive over-allocation factor controls the per-round retry rate: 4× expected "
    "for bits ≤ 48 (one-shot), 2× for bits=52, 1× for bits ≥ 56 (≈2 rounds on "
    "average to find a collision)."
)
body(
    "Code: Final/Code/task4.cu (uses gpu_hashes_narrow.cuh for shared device hash "
    "functions). Output: Final/Data/task4/scaling_task4_device.dat and "
    "scaling_task4_unified.dat. Plots: task4_compare.pdf, task4_speedup.pdf."
)

heading('3.4.1 Methodology — Why Multiple Trials Matter', level=2)
body(
    "Initial Task 4 results were single-trial, which produced misleading "
    "conclusions. The birthday-paradox-driven retry distribution "
    "(geometric) has variance comparable to its mean — at bits=56 with 1× "
    "over-allocation, a single run can take anywhere from 1 to 6+ rounds "
    "depending purely on which input value happens to collide first. We "
    "re-ran with 10 trials per configuration in BOTH device and unified "
    "modes and report median ± IQR. Device mode also has small variance "
    "(~1.8% miss rate at 4× allocation, plus 5–10% jitter from GPU clock "
    "boost and kernel launch overhead); since device-mode trials are "
    "sub-second, repetition is free. Pollard's rho is essentially "
    "deterministic given fixed seeds and thread scheduling, so a single "
    "trial suffices for Task 3 comparisons."
)

heading('3.4.2 Finding — Unified Memory Has Real but Variable Overhead at Small Bits', level=2)
body(
    "Comparing unified vs device mode at bit widths where the working set "
    "fits in 8 GB VRAM (no page migration required):"
)
add_table(
    ['bits', 'MD5 unified/device', 'SHA-1 unified/device', 'SHA-256 unified/device'],
    [
        ['24',  '3.7×', '4.0×',   '3.7×'],
        ['32',  '2.3×', '0.6× *', '0.6× *'],
        ['40',  '2.2×', '1.6×',   '1.8×'],
        ['48',  '2.0×', '2.3×',   '2.4×'],
    ]
)
body(
    "* SHA-1 and SHA-256 at bits=32 are sub-millisecond; the ratio is "
    "dominated by measurement noise. At meaningful problem sizes (bits=48) "
    "the page-tracking overhead is consistently 2.0–2.4×. Conclusion: "
    "Unified Memory has a real ~2× tax even with no spilling; it is a "
    "tool for oversubscription, not a free abstraction."
)

heading('3.4.3 Finding — Per-round Thrust Beats Pollard\'s ρ; Total Time Depends on Retry Luck', level=2)
body(
    "At bits=56 the comparison against Pollard's rho is more nuanced than "
    "single-trial data suggested:"
)
add_table(
    ['Algorithm (bits=56)', 'Pollard\'s ρ (ms)', 'Thrust uni total (ms)', 'rounds', 'per-round (ms)'],
    [
        ['MD5',      '2,133',  '3,359',  '6.0', '560'],
        ['SHA-1',    '2,375',  '1,161',  '1.0', '1,161'],
        ['SHA-256',  '1,930',  '1,951',  '3.0', '650'],
    ]
)
body(
    "Per-round Thrust unified time (560–1,161 ms) is consistently FASTER "
    "than Pollard's rho's deterministic single-run time (1,930–2,375 ms). "
    "However, total wall time depends on how many retry rounds are needed "
    "— SHA-1 got lucky in 1 round (2× faster than Pollard), MD5 needed 6 "
    "(1.6× slower), SHA-256 needed 3 (tied). The 10-trial median (see "
    "task4_trial_spread.pdf) collapses this variance and shows the two "
    "algorithms are in the same performance class at bits=56. Pollard's "
    "advantage is determinism, not raw speed."
)

heading('3.4.4 Finding — PCIe Spill Boundary is Sharp at bits 52→56', level=2)
add_table(
    ['bits', 'MD5 unified median (ms)', 'factor vs prior', 'theoretical 2^(bits/2)×'],
    [
        ['48',     '240.9',  '—',     '—'],
        ['52',     '516.4',  '2.1×',  '2.0× (still fits in VRAM)'],
        ['56',   '3,359',    '6.5×',  '4.0× (extra 1.6× = PCIe spill)'],
    ]
)
body(
    "From bits=48 → 52 the slowdown matches theory exactly — sort scratch "
    "still fits in VRAM. From bits=52 → 56 the extra 1.6× factor beyond "
    "theoretical scaling is the PCIe-bound sort overhead. The 10-trial "
    "bits=64 measurement is in progress; results will be added when the "
    "rerun completes."
)
body("GPU algorithm selection guide based on measured evidence:")
bullet("bits ≤ 48: Thrust device-mode is fastest absolute (5–35× over Pollard's ρ)")
bullet("bits 52: Both Thrust modes work; unified pays ~2× tax but still beats Pollard's ρ")
bullet("bits 56: Comparable performance class — choose by deterministic-time preference")
bullet("bits ≥ 72: Thrust working set exceeds 96 GB host RAM — Pollard's rho only feasible option")

# ── 4. Deliverables ───────────────────────────────────────────────────────────
heading('4. Deliverables: Building and Running')

heading('4.1 Repository Structure', level=2)
body("All project code is in the Final/ subdirectory of the repo:")
bullet("Final/Code/       — task1.cu, task2.cu, task3.cu, md5/sha1/sha256 CPU sources")
bullet("Final/sbatch/     — SLURM sbatch scripts for all tasks (task1/, task2/, task3/)")
bullet("Final/Scripts/    — Python plotting scripts (plot_final_task*.py)")
bullet("Final/Data/       — Output .dat files and generated PDF/PNG plots")
bullet("Final/Report/     — task1/2/3 analysis text files and this report")

heading('4.2 Compilation', level=2)
body("All commands run from the repository root on the Euler cluster after: "
     "module load nvidia/cuda/13.0.0")
body("Task 1 and Task 2:")
doc.add_paragraph(
    "nvcc -O3 -std=c++17 -o Final/task1 Final/Code/task1.cu "
    "Final/Code/md5_cpu.cpp Final/Code/sha1_cpu.cpp Final/Code/sha256_cpu.cpp\n"
    "nvcc -O3 -std=c++17 -o Final/task2 Final/Code/task2.cu "
    "Final/Code/md5_cpu.cpp Final/Code/sha1_cpu.cpp Final/Code/sha256_cpu.cpp",
    style='Normal'
).runs[0].font.name = 'Courier New'

body("Task 3 (requires -Xcompiler -fopenmp for OpenMP support):")
doc.add_paragraph(
    "nvcc -O3 -std=c++17 -Xcompiler -fopenmp -o Final/task3 Final/Code/task3.cu "
    "Final/Code/md5_cpu.cpp Final/Code/sha1_cpu.cpp Final/Code/sha256_cpu.cpp",
    style='Normal'
).runs[0].font.name = 'Courier New'

body("Task 4 (Thrust ships with CUDA — no extra link flags needed):")
doc.add_paragraph(
    "nvcc -O3 -std=c++17 -o Final/task4 Final/Code/task4.cu",
    style='Normal'
).runs[0].font.name = 'Courier New'

heading('4.3 Running via SLURM', level=2)
body("Submit all three Task 3 scaling jobs (one per algorithm):")
doc.add_paragraph(
    "sbatch Final/sbatch/task3/task3_md5_scaling.sh\n"
    "sbatch Final/sbatch/task3/task3_sha1_scaling.sh\n"
    "sbatch Final/sbatch/task3/task3_sha256_scaling.sh",
    style='Normal'
).runs[0].font.name = 'Courier New'
body("SLURM parameters: -p instruction, --gres=gpu:1, --cpus-per-task=12, "
     "--mem=96G, -t 0-00:40:00. Data written to Final/Data/task3/.")

heading('4.4 Generating Plots', level=2)
doc.add_paragraph(
    "python3 Final/Scripts/task1/plot_final_task1.py\n"
    "python3 Final/Scripts/task2/plot_final_task2.py\n"
    "python3 Final/Scripts/task3/plot_final_task3.py\n"
    "python3 Final/Scripts/task3/plot_omp_scaling.py",
    style='Normal'
).runs[0].font.name = 'Courier New'

# ── 5. Conclusions ────────────────────────────────────────────────────────────
heading('5. Conclusions and Future Work')

body(
    "This project demonstrates three distinct regimes of GPU advantage for hash "
    "computations:"
)
bullet(
    "Raw throughput (Tasks 1–2): The GPU achieves 167–440× throughput over serial "
    "CPU for large batches, driven by thread-level parallelism over embarrassingly "
    "parallel independent hashes. Register pressure determines per-algorithm sensitivity "
    "to configuration, and tpb=32 is universally optimal for register-heavy hash kernels."
)
bullet(
    "Parallel CPU ceiling (Task 3, OpenMP): OpenMP parallelization of collision search "
    "is limited to ~2× speedup over serial at large bit widths due to the serial merge "
    "phase (Amdahl's Law) and NUMA topology. The 12-thread sweet spot is hardware-"
    "specific to Euler's dual-socket node and would differ on NUMA-free hardware."
)
bullet(
    "Algorithmic advantage (Task 3, GPU): The GPU's decisive advantage at bits≥48 is "
    "not just raw speed but algorithmic — Pollard's rho with distinguished points uses "
    "O(1) memory per thread, while any birthday-paradox CPU approach requires O(2^(N/2)) "
    "memory. At bits=64 the CPU needs ~100 GB; the GPU uses 8 MB. This makes bits=64–72 "
    "GPU-only regardless of CPU clock speed or parallelism level."
)
bullet(
    "Empirical confirmation via Task 4 (10-trial median methodology): With Unified "
    "Memory exposing host RAM to the GPU, the Thrust sort approach falls into the "
    "same performance class as Pollard's rho at bits=56 — per-round Thrust is faster, "
    "but retry variance makes total time comparable. Pollard's rho's advantage at "
    "bits=56 is determinism and predictable timing, not raw speed. At bits ≥ 72 the "
    "Thrust working set exceeds available host RAM and Pollard's rho is the only "
    "feasible option. The lesson: O(1)-memory algorithms are valuable above the VRAM "
    "boundary not just for feasibility, but for predictability and graceful scaling."
)

body("ME759 concepts leveraged in this project:")
bullet("CUDA kernel design: template specialization for zero warp divergence, "
       "cudaEvent_t timing, cudaMalloc/cudaFree memory management")
bullet("Occupancy and register pressure: SM register file limits, occupancy "
       "analysis, tpb tuning for register-heavy kernels")
bullet("OpenMP: parallel-for with schedule(static), omp_get_max_threads(), "
       "phase-parallel/serial-merge pattern")
bullet("NUMA awareness: dual-socket topology, cross-socket memory latency, "
       "thread affinity and sweet spot analysis")
bullet("Amdahl's Law: serial fraction identification, theoretical speedup ceiling "
       "calculation for the two-phase parallel algorithm")

body("Future work:")
bullet(
    "bits=80 GPU: achievable in ~30 minutes but exceeds Euler's 40-minute instruction "
    "partition limit. Running on a partition with longer wall-clock limits would extend "
    "the speedup curve one more data point to bits=80."
)
bullet(
    "Parallel sort-based CPU collision search: replacing unordered_map with a flat "
    "sorted array reduces memory to ~43 GB at bits=64. A fully parallel radix sort + "
    "linear scan could reach bits=64 on CPU in 20–30 seconds on a high-memory server, "
    "narrowing the algorithmic gap with Pollard's rho."
)

body(
    "Implemented Task 4 — GPU Thrust Sort variant — and proposed extension: "
    "Task 4's Thrust-based GPU collision search is implemented and benchmarked "
    "(see Section 3.4); it complements Pollard's rho by using sort_by_key + "
    "adjacent-duplicate scan, demonstrating where each algorithm provides maximum "
    "speedup as a function of bit width. A natural further extension is the third "
    "GPU strategy not yet implemented:"
)
bullet(
    "GPU Birthday Attack with device-side concurrent hash table. Allocate a large "
    "open-addressing hash table directly in GPU VRAM and have all 65,536 threads "
    "insert (truncated_hash → index) pairs in parallel using atomicCAS for lock-free "
    "collision detection. No CPU-side merge step is needed — the collision is detected "
    "on-device the moment two threads map to the same slot. "
    "Expected behavior: at small bit widths (bits≤48) where the entire table fits in "
    "GPU VRAM (8 GB GPU holds 2^29 × 8B = 4 GB), this approach should be faster than "
    "Pollard's rho because it eliminates the chain-walking overhead and the DP "
    "detection round-trips. At bits≥56 the table exceeds VRAM and Pollard's rho "
    "with O(1) memory wins. The crossover point is the main finding. "
    "Implementation: a flat uint64_t[2^(bits/2+2)] device array, atomicCAS for "
    "insert, a single __global__ kernel with early-exit via a device-side flag."
)
bullet(
    "GPU Thrust Parallel Sort. Hash a full batch of N = sqrt(pi/2) * 2^(bits/2) "
    "inputs on GPU, store (hash, index) pairs in device memory, then use "
    "thrust::sort_by_key to sort by hash, and a final kernel to scan for adjacent "
    "equal-hash pairs. This is entirely GPU-resident — no CPU merge bottleneck — "
    "and directly parallelizes the sort-based approach that would improve CPU "
    "performance. "
    "Expected behavior: faster than Pollard's rho at small bits (sort is O(N log N) "
    "but fully parallel and cache-friendly); slower at large bits where N exceeds "
    "VRAM. Thrust radix sort on GPU achieves ~1 GB/s for 64-bit keys, placing bits=48 "
    "(16M pairs × 16B = 256 MB) at ~0.3 seconds sort time, well under Pollard's "
    "544 ms. The crossover with Pollard's rho is expected around bits=52–56, where "
    "the Thrust sort working set saturates VRAM. "
    "Implementation: thrust::device_vector<thrust::pair<uint64_t,uint64_t>>, "
    "thrust::sort_by_key, then thrust::adjacent_difference to flag duplicates."
)
body(
    "Together, Tasks 3 and 4 would establish a complete GPU algorithm selection guide: "
    "Thrust sort wins at bits≤50, GPU birthday table wins at bits≤56 (given sufficient "
    "VRAM), and Pollard's rho with distinguished points wins at bits≥56 where O(1) "
    "memory is the only feasible approach."
)

# ── References ────────────────────────────────────────────────────────────────
heading('References')
body("[1] Rivest, R. (1992). The MD5 Message-Digest Algorithm. RFC 1321.")
body("[2] NIST FIPS PUB 180-4: Secure Hash Standard (SHS). 2015.")
body("[3] van Oorschot, P. C., & Wiener, M. J. (1999). Parallel Collision Search with "
     "Cryptanalytic Applications. Journal of Cryptology, 12(1), 1–28. "
     "(Pollard's rho with distinguished points.)")
body("[4] Kirk, D. B., & Hwu, W.-m. W. (2016). Programming Massively Parallel "
     "Processors: A Hands-on Approach (3rd ed.). Morgan Kaufmann.")
body("[5] OpenMP Architecture Review Board. OpenMP Application Programming Interface, "
     "Version 5.2. 2021.")
body("[6] NVIDIA Corporation. (2024). CUDA C++ Programming Guide. "
     "https://docs.nvidia.com/cuda/cuda-c-programming-guide/ "
     "(cudaEvent_t timing, __constant__ memory, atomicAdd, warp scheduling, "
     "register file occupancy.)")
body("[7] Pollard, J. M. (1978). Monte Carlo methods for index computation mod p. "
     "Mathematics of Computation, 32(143), 918–924. "
     "(Original rho cycle-detection algorithm; extended by van Oorschot & Wiener [3].)")
body("[8] Flajolet, P., & Odlyzko, A. (1990). Random mapping statistics. "
     "Advances in Cryptology – EUROCRYPT '89, Lecture Notes in Computer Science, "
     "vol. 434, pp. 329–354. Springer. "
     "(Source for birthday paradox collision expectation E ≈ √(π/2) · 2^(N/2).)")
body("[9] Bell, N., & Hoberock, J. (2011). Thrust: A productivity-oriented library "
     "for CUDA. GPU Computing Gems Jade Edition, pp. 359–371. Morgan Kaufmann. "
     "(Referenced in Task 4 proposal for GPU parallel sort-based collision search.)")
body("[10] Anthropic. (2026). Claude Code with Claude Sonnet 4.6 (claude-sonnet-4-6) "
     "[AI coding assistant]. Used for code generation, algorithm implementation, and "
     "analysis throughout this project. https://www.anthropic.com/claude-code")

out = 'Final/Report/FinalProject_Arsov.docx'
os.makedirs('Final/Report', exist_ok=True)
doc.save(out)
print(f"Saved {out}")
