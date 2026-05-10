import os
import time

import cv2
import numpy as np

from fpie import core_openmp
from fpie.io import read_images, write_image
from fpie.process import GridProcessor

import config


def main():
    # Initialize LIKWID marker API (must be before any markers)
    core_openmp.likwid_init()

    # 1. Load images
    print(f"Loading images from {config.TEST_PREFIX}...")
    try:
        src, mask, tgt = read_images(config.SRC_PATH, config.MASK_PATH, config.TGT_PATH)
    except Exception as e:
        print(f"Error loading images: {e}")
        print(f"Please ensure {config.SRC_PATH}, {config.MASK_PATH}, and {config.TGT_PATH} exist.")
        return

    # 2. Initialize the OpenMP Grid Processor
    # You can change backend to 'openmp' and specify number of CPUs
    print("Initializing OpenMP solver...")
    proc = GridProcessor(
        gradient="src",
        backend="openmp",
        n_cpu=config.N_CPU,  # Adjust based on your hardware
        grid_x=config.GRID_X,
        grid_y=config.GRID_Y,
    )

    # 3. Reset the solver with input data
    # (0, 0) are the offsets for mask on src and (130, 130) for mask on tgt
    n_vars = proc.reset(src, mask, tgt, config.SRC_OFFSET, config.TGT_OFFSET)
    print(f"Number of variables to solve: {n_vars}")

    # 4. Run solver iterations
    print("Running solver...")
    start_time = time.time()

    # Run 5000 iterations in chunks of 100
    total_iters = 5000
    chunk_size = 100
    result = tgt

    for i in range(0, total_iters, chunk_size):
        result, err = proc.step(chunk_size)
        print(f"Iteration {i + chunk_size}, Error: {err}")

    end_time = time.time()
    print(f"Solved in {end_time - start_time:.4f} seconds")

    # 5. Save the result
    write_image("result2.jpg", result)
    print("Saved result to result2.jpg")

    # Flush LIKWID marker data (must be before exit)
    core_openmp.likwid_close()


if __name__ == "__main__":
    main()
