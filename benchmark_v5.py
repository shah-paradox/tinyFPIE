import os
import time

import cv2
import numpy as np

import config
from fpie import core_openmp
from fpie.io import read_images, write_image
from fpie.process import GridProcessor


def main():
    # Initialize LIKWID marker API (must be before any markers)
    core_openmp.likwid_init()

    # 1. Load images
    print(f"Loading images from {config.TEST_PREFIX}...")
    try:
        src, mask, tgt = read_images(config.SRC_PATH, config.MASK_PATH, config.TGT_PATH)
    except Exception as e:
        print(f"Error loading images: {e}")
        print(
            f"Please ensure {config.SRC_PATH}, {config.MASK_PATH}, and {config.TGT_PATH} exist."
        )
        return

    # 2. Initialize the OpenMP Grid Processor and swap the core for SolverV5
    print("Initializing OpenMP Solver v5...")
    proc = GridProcessor(
        gradient="max",
        backend="openmp",
        n_cpu=config.N_CPU,
        grid_x=config.GRID_X,
        grid_y=config.GRID_Y,
    )
    # Swap the backend core for the new SolverV5
    proc.core = core_openmp.SolverV5(config.GRID_X, config.GRID_Y, config.N_CPU)

    # 3. Reset the solver with input data
    n_vars = proc.reset(src, mask, tgt, config.SRC_OFFSET, config.TGT_OFFSET)
    print(f"Number of variables to solve: {n_vars}")

    # 4. Run solver iterations
    print("Running solver...")
    start_time = time.time()

    # Run 970 iterations in chunks of 100
    total_iters = 1000
    chunk_size = 100
    result = tgt

    for i in range(0, total_iters, chunk_size):
        result, err = proc.step(chunk_size)
        print(f"Iteration {i + chunk_size}, Error: {err}")

    end_time = time.time()
    print(f"Solved in {end_time - start_time:.4f} seconds")

    # 5. Save the result
    write_image("result_v5.jpg", result)
    print("Saved result to result_v5.jpg")

    # Flush LIKWID marker data (must be before exit)
    core_openmp.likwid_close()


if __name__ == "__main__":
    main()
