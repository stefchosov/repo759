"""
Generates Final/Report/FinalProject_Arsov.docx from template structure.
Run from repo root: python3 Final/Scripts/generate_report.py
"""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

doc = Document()

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

def figure(path, caption, width_inches=6.0):
    """Embed a PNG with a centered caption underneath."""
    if not os.path.exists(path):
        body(f"[figure missing: {path}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(path, width=Inches(width_inches))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(10)

# Title page
title = doc.add_heading('Spring 2026 ME/CS/ECE 759 Final Project Report', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph('University of Wisconsin-Madison')
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

proj = doc.add_paragraph('CPU vs GPU Performance and Truncated-Collision Analysis\nof MD5, SHA-1, and SHA-256')
proj.alignment = WD_ALIGN_PARAGRAPH.CENTER
proj.runs[0].bold = True

author = doc.add_paragraph('Stefan Arsov\nMay 5, 2026')
author.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_page_break()

# Abstract
heading('Abstract')
body(
    "This project benchmarks MD5, SHA-1, and SHA-256 across CPU serial, OpenMP, and "
    "CUDA, and uses truncated-collision search to identify where the GPU's algorithmic "
    "advantage becomes decisive. Task 1 measures raw hash throughput vs batch size, "
    "establishing GPU vs CPU baselines (up to 440x for SHA-1 at n=10M). Task 2 sweeps "
    "threads-per-block and finds tpb=32 optimal for all three algorithms because of "
    "register pressure. Task 3 implements collision search across bit widths {16..72} "
    "with three approaches (CPU serial, OpenMP-12, GPU Pollard's rho with distinguished "
    "points). The GPU is the only implementation that runs at bits >= 64, and at "
    "bits=56 it achieves 260x over CPU serial for MD5. A 128-bit extension lets the "
    "GPU reach bits=72. Task 4 adds a second GPU strategy (Thrust parallel sort) in "
    "two memory modes (device-only and unified) with 10-trial median methodology, and "
    "shows that above the VRAM boundary Pollard's rho's main advantage is determinism "
    "rather than raw speed. All experiments ran on the UW-Madison Euler cluster (NVIDIA "
    "GPU, CUDA 13.0, dual-socket CPU node, 96 GB host RAM)."
)
body("Link to Final Project git repo: https://github.com/stefchosov/repo759")
doc.add_page_break()

# General information
heading('General Information')
bullet("Name: Stefan Arsov")
bullet("Email: sarsov@epic.com")
bullet("Home department: Computer Sciences")
bullet("Status: MS Student")
bullet("Teammate: None")
bullet("I release the ME759 Final Project code as open source and under a BSD3 license "
       "for unfettered use of it by any interested party.")

# 1. Problem Statement
heading('1. Problem Statement')
body(
    "Cryptographic hash functions are the computational backbone of password hashing, "
    "digital signatures, file integrity verification, and certificate chains. Their "
    "security rests on preimage and collision resistance. While the full-length "
    "versions are cryptographically strong, truncated variants are common in practice: "
    "SSH host key fingerprints use truncated MD5 or SHA-256, Git object IDs use "
    "truncated SHA-1, and many authentication protocols use only the first N bits to "
    "reduce bandwidth."
)
body(
    "This project asks two questions. First, how does GPU parallelism affect raw hash "
    "throughput relative to CPU implementations, and how does it vary by algorithm "
    "and configuration? Second, at what bit width does GPU parallelism make truncated "
    "collision search possible where CPU approaches break down? The motivation is to "
    "draw a clear line between CPU-feasible and GPU-only cryptanalysis tasks, which "
    "is directly relevant to password cracking, rainbow table construction, and "
    "certificate spoofing."
)

# 2. Solution Description
heading('2. Solution Description')

heading('2.1 Code Structure', level=2)
body("All code lives under Final/Code/ in the repository:")
bullet("task1.cu: launches n parallel hash kernels, cudaEvent_t timing, reports MH/s.")
bullet("task2.cu: same kernel, sweeps tpb in {32, 64, 128, 256, 512, 1024}.")
bullet("task3.cu: CPU serial, OpenMP, and GPU Pollard's rho collision search.")
bullet("task4.cu: GPU Thrust-sort collision search, device and unified memory modes.")
bullet("gpu_hashes_narrow.cuh: shared device hash functions for Task 4.")
bullet("md5_cpu.cpp / sha1_cpu.cpp / sha256_cpu.cpp: CPU reference implementations.")

heading('2.2 Tasks 1 and 2: Throughput and tpb Sweep', level=2)
body(
    "Task 1 launches n CUDA threads, each computing one hash of an 8-byte little-endian "
    "encoding of its thread index. Three template kernels (template<int ALGO>) "
    "specialize at compile time so there are no runtime branches inside the kernels, "
    "and the GPU hash functions operate entirely in registers. Timing uses cudaEvent_t "
    "around the kernel call. The experiment sweeps n from 100K to 10M to find the "
    "GPU break-even point relative to serial CPU."
)
body(
    "Task 2 holds n=10,000,000 fixed and sweeps tpb to test occupancy effects. The "
    "hash kernels are register-heavy (SHA-256 keeps 9 32-bit working variables plus "
    "a 64-entry message schedule), so larger blocks reduce concurrent warp count "
    "and hide less latency."
)

heading('2.3 Task 3: Truncated Collision Search', level=2)
body(
    "Task 3 finds two distinct inputs i and j whose N-bit truncated hashes match. "
    "Three approaches are implemented and compared:"
)
bullet(
    "CPU serial: sequential scan inserting each truncated hash into "
    "std::unordered_map<uint64_t, uint64_t>; first duplicate key is the collision. "
    "Limited to bits <= 56 due to memory (about 17 GB at bits=56)."
)
bullet(
    "OpenMP (12 threads): two-phase per batch. Phase 1 is a parallel hash into "
    "pre-allocated buffers (no synchronization). Phase 2 is a serial merge into the "
    "global map, stopping at the first duplicate. Thread count is fixed at 12 because "
    "a dedicated scaling experiment from 1 to 32 threads showed peak performance there "
    "(NUMA boundary on the dual-socket node, plus Amdahl's serial merge ceiling and "
    "L3 cache pressure beyond 12 threads)."
)
bullet(
    "GPU Pollard's rho with distinguished points: 65,536 CUDA threads each walk an "
    "independent Markov chain x_{n+1} = truncate(hash(x_n), N). A 'distinguished "
    "point' is any chain value with its low N/4 bits zero. Two chains reaching the "
    "same DP from different starts imply a collision. Uses O(1) memory per thread "
    "(8 MB DP buffer total). A 128-bit extension (uint128 struct, wide hash functions, "
    "pollard_kernel_wide template) enables bits=72."
)

heading('2.4 Task 4: GPU Algorithm Comparison via Thrust Sort', level=2)
body(
    "Task 4 implements a second GPU collision search and benchmarks it against Task 3's "
    "Pollard's rho. Three phases per round: a parallel hash kernel writes (hash, index) "
    "pairs to device arrays, thrust::sort_by_key sorts by hash, and a parallel "
    "find-duplicate kernel uses atomicMin to record the smallest input index where two "
    "adjacent sorted hashes match. If no duplicate is found, a new round retries with "
    "a fresh starting offset."
)
body(
    "Two memory modes are supported. Device mode uses cudaMalloc and is capped at "
    "bits=48 because the working set saturates the 8 GB GPU at bits >= 52. Unified "
    "memory mode uses cudaMallocManaged so buffers spill into host RAM via PCIe "
    "page migration, extending the sweep to bits=52 and bits=56 at the cost of "
    "PCIe-bound sort performance. An adaptive over-allocation factor (4x at bits "
    "<= 48, 2x at bits=52, 1x at bits=56) controls the find rate per round. Each "
    "configuration runs 10 trials and the median is reported, because the geometric "
    "retry distribution at sub-1x allocation has variance comparable to its mean."
)

# 3. Results
heading('3. Overview of Results')

heading('3.1 Task 1: Throughput', level=2)
add_table(
    ['Algorithm', 'CPU Serial (MH/s)', 'OpenMP-4 (MH/s)', 'GPU (MH/s)', 'GPU/Serial'],
    [
        ['MD5',     '6.53',  '22.73', '2694.6', '413x'],
        ['SHA-1',   '2.78',   '9.34', '1224.8', '440x'],
        ['SHA-256', '3.94',   '9.54',  '658.9', '167x'],
    ]
)
figure('Final/Data/task1/task1_throughput.png',
       'Figure 1. Hash throughput vs batch size n. GPU breaks even with serial CPU '
       'around n = 200,000.')
body(
    "OpenMP-4 achieves near-linear 3.5x speedup for MD5 and SHA-1, and 2.4x for "
    "SHA-256 (limited by register pressure). Below n ≈ 200,000 the GPU is slower "
    "than serial because the ~2 ms kernel launch overhead dominates."
)

heading('3.2 Task 2: Threads-per-Block', level=2)
add_table(
    ['tpb', 'MD5 (MH/s)', 'SHA-1 (MH/s)', 'SHA-256 (MH/s)'],
    [
        ['32 (optimal)',  '3021.1', '1201.9', '574.7'],
        ['256 (default)', '2277.9', '1021.5', '537.3'],
        ['1024',          '2375.3', '1003.0', '557.4'],
    ]
)
figure('Final/Data/task2/task2_tpb.png',
       'Figure 2. GPU throughput vs threads-per-block. tpb=32 is optimal for all '
       'three algorithms because larger blocks reduce SM occupancy.')
body(
    "MD5 drops 27% from tpb=32 to tpb=512 (most sensitive, since it starts with "
    "highest occupancy). SHA-256 drops only 6% (already register-starved). The "
    "default tpb=256 leaves a free 7% to 33% on the table."
)

heading('3.3 Task 3: Collision Search', level=2)
add_table(
    ['Algo', 'Bits', 'CPU Serial (ms)', 'OMP-12 (ms)', 'GPU (ms)', 'GPU/CPU'],
    [
        ['MD5',     '32', '9.5',     '2.6',     '36.7',  '0.26x'],
        ['MD5',     '48', '4,234',   '2,327',   '544',   '7.8x'],
        ['MD5',     '56', '554,672', '359,864', '2,133', '260x'],
        ['MD5',     '64', 'n/a',     'n/a',     '8,488', 'n/a'],
        ['SHA-1',   '56', '74,417',  '35,765',  '2,375', '31x'],
        ['SHA-256', '56', '316,526', '181,146', '1,930', '164x'],
    ]
)
figure('Final/Data/task3/task3_time.png',
       'Figure 3. Time to find a collision vs truncated bit width. GPU is the only '
       'implementation viable at bits >= 64.')
figure('Final/Data/task3/task3_speedup.png',
       'Figure 4. GPU and OpenMP speedup over CPU serial. The GPU advantage grows '
       'with bit width.')
figure('Final/Data/task3/task3_birthday.png',
       'Figure 5. Birthday-paradox validation: cpu_count divided by expected count. '
       'Ratios near 1.0 confirm correct truncated-hash extraction.')
body(
    "GPU speedup grows with bit width because CPU memory requirements scale as "
    "2^(N/2) while the GPU's Pollard's rho uses O(1) memory per thread. OpenMP only "
    "achieves 1.5x to 2.1x at bits=56 because Phase 2 (the serial merge) becomes the "
    "bottleneck and NUMA penalties hit beyond a single socket."
)
figure('Final/Data/task3/omp_scaling_speedup.png',
       'Figure 6. OpenMP speedup vs thread count. Peak at 12 threads (single socket); '
       'degradation past 12 due to NUMA boundary and serial-merge ceiling.')

heading('3.4 Task 4: GPU Algorithm Comparison', level=2)
figure('Final/Data/task4/task4_compare.png',
       'Figure 7. Time vs bits for Pollard\'s rho, Thrust device-mode, and Thrust '
       'unified-mode. Thrust device wins where it can run; Thrust unified extends to '
       'bits=56 by spilling to host RAM.')
body(
    "At bits <= 48, Thrust device-mode is fastest absolute (5x to 35x over Pollard's "
    "rho). The two algorithms become comparable at bits=52, and at bits=56 they fall "
    "into the same performance class. Pollard's rho is the only viable algorithm at "
    "bits >= 64."
)
figure('Final/Data/task4/task4_overhead.png',
       'Figure 8. Unified vs device memory overhead at bits where the working set '
       'still fits in VRAM. The page-tracking driver tax is a consistent 2x to 2.4x.')
body(
    "Even with no spilling, unified memory has a real cost from CUDA's "
    "page-residency tracking. Unified memory is a tool for oversubscription, not a "
    "free abstraction."
)
figure('Final/Data/task4/task4_trial_spread.png',
       'Figure 9. Per-trial spread at bits=52 and bits=56 (10 trials each). The '
       'green dashed line is Pollard\'s rho\'s deterministic time. Thrust unified '
       'has high variance from the geometric retry distribution.')
body(
    "Each dot is one of 10 trials; the purple bar marks the median. At bits=56 some "
    "trials beat Pollard's rho and some lose, depending purely on retry luck. "
    "Pollard's rho's advantage above the VRAM boundary is determinism, not raw "
    "speed. bits=64 was attempted in unified mode but every trial crashed inside "
    "thrust::sort_by_key during temp-buffer allocation, so it is dropped from the "
    "sweep. This reinforces the Task 3 conclusion: at bits >= 64, Pollard's rho is "
    "the only approach that runs at all."
)

# 4. Deliverables
heading('4. Deliverables: Building and Running')

heading('4.1 Repository Structure', level=2)
bullet("Final/Code/: task1.cu, task2.cu, task3.cu, task4.cu, gpu_hashes_narrow.cuh, "
       "and CPU hash sources.")
bullet("Final/sbatch/: SLURM scripts for all tasks.")
bullet("Final/Scripts/: Python plot generators.")
bullet("Final/Data/: .dat output files and PDF/PNG plots.")
bullet("Final/Report/: per-task analyses and this report.")

heading('4.2 Compilation and Run', level=2)
body("From the repository root after `module load nvidia/cuda/13.0.0`:")
doc.add_paragraph(
    "nvcc -O3 -std=c++17 -Xcompiler -fopenmp -o Final/task3 Final/Code/task3.cu \\\n"
    "    Final/Code/md5_cpu.cpp Final/Code/sha1_cpu.cpp Final/Code/sha256_cpu.cpp\n"
    "nvcc -O3 -std=c++17 -o Final/task4 Final/Code/task4.cu",
    style='Normal'
).runs[0].font.name = 'Courier New'

body("Submit jobs:")
doc.add_paragraph(
    "sbatch Final/sbatch/task3/task3_md5_scaling.sh   # also sha1, sha256\n"
    "sbatch Final/sbatch/task4/task4_scaling.sh",
    style='Normal'
).runs[0].font.name = 'Courier New'
body("Task 3 SLURM parameters: -p instruction, --gres=gpu:1, --cpus-per-task=12, "
     "--mem=96G, -t 0-00:40:00.")

body("Generate plots:")
doc.add_paragraph(
    "python3 Final/Scripts/task1/plot_final_task1.py\n"
    "python3 Final/Scripts/task2/plot_final_task2.py\n"
    "python3 Final/Scripts/task3/plot_final_task3.py\n"
    "python3 Final/Scripts/task3/plot_omp_scaling.py\n"
    "python3 Final/Scripts/task4/plot_final_task4.py",
    style='Normal'
).runs[0].font.name = 'Courier New'

# 5. Conclusions
heading('5. Conclusions and Future Work')
body("Three regimes of GPU advantage emerge across the four tasks:")
bullet(
    "Raw throughput (Tasks 1 and 2): GPU is 167x to 440x faster than CPU serial for "
    "large batches. tpb=32 is universally optimal for register-heavy kernels."
)
bullet(
    "Parallel CPU ceiling (Task 3 OpenMP): only 1.5x to 2.1x speedup at bits=56 due "
    "to Amdahl's serial merge and NUMA. The 12-thread sweet spot is hardware-specific "
    "to Euler's dual-socket node."
)
bullet(
    "Algorithmic advantage (Task 3 GPU): O(1) memory per thread via Pollard's rho "
    "makes bits=64 to 72 GPU-only. At bits=64 the CPU needs roughly 100 GB; the GPU "
    "uses 8 MB."
)
bullet(
    "Memory hierarchy and oversubscription (Task 4): even with unified memory and "
    "96 GB host RAM, Thrust sort merely matches Pollard's rho at bits=56 and crashes "
    "outright at bits=64. Above the VRAM boundary the GPU advantage is determinism "
    "and graceful scaling, not raw bandwidth."
)

body("ME759 concepts applied:")
bullet("CUDA template kernels for zero warp divergence and cudaEvent_t timing")
bullet("Occupancy and register pressure analysis")
bullet("OpenMP parallel-for and phase-parallel/serial-merge patterns")
bullet("NUMA topology and dual-socket awareness")
bullet("Amdahl's Law applied to a real two-phase algorithm")
bullet("GPU memory hierarchy, VRAM limits, and unified-memory tradeoffs")

body("Future work:")
bullet(
    "bits=80 GPU: feasible in roughly 30 minutes but exceeds Euler's 40-minute "
    "instruction-partition wall-clock limit."
)
bullet(
    "Parallel sort-based CPU collision search: replacing unordered_map with a flat "
    "sorted array could reach bits=64 on a high-memory CPU server in 20-30 seconds."
)
bullet(
    "GPU device-side concurrent hash table (atomicCAS open-addressing): the third "
    "GPU strategy from the original Task 4 proposal, expected to beat Pollard's rho "
    "at small bit widths and lose at large ones."
)

# References
heading('References')
body("[1] Rivest, R. (1992). The MD5 Message-Digest Algorithm. RFC 1321.")
body("[2] NIST FIPS PUB 180-4: Secure Hash Standard (SHS). 2015.")
body("[3] van Oorschot, P. C., and Wiener, M. J. (1999). Parallel Collision Search "
     "with Cryptanalytic Applications. Journal of Cryptology, 12(1), 1-28.")
body("[4] Kirk, D. B., and Hwu, W.-m. W. (2016). Programming Massively Parallel "
     "Processors: A Hands-on Approach (3rd ed.). Morgan Kaufmann.")
body("[5] OpenMP Architecture Review Board. OpenMP Application Programming Interface, "
     "Version 5.2. 2021.")
body("[6] NVIDIA Corporation. (2024). CUDA C++ Programming Guide. "
     "https://docs.nvidia.com/cuda/cuda-c-programming-guide/")
body("[7] Pollard, J. M. (1978). Monte Carlo methods for index computation mod p. "
     "Mathematics of Computation, 32(143), 918-924.")
body("[8] Flajolet, P., and Odlyzko, A. (1990). Random mapping statistics. "
     "EUROCRYPT '89, LNCS vol. 434, pp. 329-354. Springer.")
body("[9] Bell, N., and Hoberock, J. (2011). Thrust: A productivity-oriented library "
     "for CUDA. GPU Computing Gems Jade Edition, pp. 359-371. Morgan Kaufmann.")
body("[10] Anthropic. (2026). Claude Code with Claude Sonnet 4.6 (claude-sonnet-4-6) "
     "[AI coding assistant]. Used for code generation, algorithm implementation, and "
     "analysis. https://www.anthropic.com/claude-code")

out = 'Final/Report/FinalProject_Arsov.docx'
os.makedirs('Final/Report', exist_ok=True)
doc.save(out)
print(f"Saved {out}")
