#!/usr/bin/env python3
"""
Benchmark runner for likwid-perfctr.
Runs benchmarks across test0..test4.
Usage: python run_benchmarks.py
"""

import os
import subprocess
from datetime import datetime

# ==================== CONFIG ====================
THREADS = 8
CPU_LIST = "0-7"
TEST_PREFIXES = ["test0", "test1", "test2", "test3", "test4"]
GRID_X = 8
GRID_Y = 8
OUTPUT_FILE = "bencher.txt"

# Benchmarks to profile with likwid (MEM + L2CACHE)
LIKWD_BENCHS = [
    "benchmark_mine.py",
    "benchmark_openmp.py",
    "benchmark_v2.py",
    "benchmark_v3.py",
    "benchmark_v4.py",
    "benchmark_v5.py",  # <-- now included
]
# ================================================


def write_config(test_prefix):
    cfg = f'''import os

# --- Performance settings ---
OMP_NUM_THREADS = "{THREADS}"
N_CPU = {THREADS}

# --- Grid settings ---
GRID_X = {GRID_X}
GRID_Y = {GRID_Y}

# --- Image paths ---
TEST_DIR = "tests"
TEST_PREFIX = "{test_prefix}"

SRC_PATH = f"{{TEST_DIR}}/{{TEST_PREFIX}}_src.png"
MASK_PATH = f"{{TEST_DIR}}/{{TEST_PREFIX}}_mask.png"
TGT_PATH = f"{{TEST_DIR}}/{{TEST_PREFIX}}_target.png"

# --- Offsets ---
SRC_OFFSET = (0, 0)
TGT_OFFSET = (0, 0)

# Apply environment variables
os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS
'''
    with open("config.py", "w") as f:
        f.write(cfg)
    print(f"[config.py] threads={THREADS}, test={test_prefix}, grid={GRID_X}x{GRID_Y}")


def run(cmd, env, label, out):
    header = f"\n{'=' * 70}\n[{datetime.now()}] {label}\n{'=' * 70}\n"
    print(header.strip())
    out.write(header)

    result = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out.write(result.stdout)
    if result.returncode != 0:
        out.write(f"\n[ERROR] exit code {result.returncode}\n")
    out.flush()

    lines = result.stdout.rstrip().splitlines()
    print("\n".join(lines[-15:]) if len(lines) > 15 else result.stdout)


def main():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS)
    env["LIKWID_FORCE"] = "1"

    with open(OUTPUT_FILE, "w") as f:
        f.write(f"Benchmark session — {datetime.now()}\n")
        f.write(
            f"Threads: {THREADS} | CPUs: {CPU_LIST} | Tests: {', '.join(TEST_PREFIXES)}\n"
        )
        f.write("=" * 70 + "\n")

    with open(OUTPUT_FILE, "a") as f:
        for test_prefix in TEST_PREFIXES:
            test_header = f"\n{'#' * 70}\n# TEST PREFIX: {test_prefix}\n{'#' * 70}\n"
            print(test_header)
            f.write(test_header)

            write_config(test_prefix)

            for script in LIKWD_BENCHS:
                for group in ("MEM", "L2CACHE"):
                    cmd = [
                        "likwid-perfctr",
                        "-C",
                        CPU_LIST,
                        "-g",
                        group,
                        "uv",
                        "run",
                        "python",
                        script,
                    ]
                    label = f"[{test_prefix}] likwid -g {group} → {script}"
                    run(cmd, env, label, f)

    print(f"\nDone. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
