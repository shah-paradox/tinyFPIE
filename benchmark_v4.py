import os
import time

import cv2
import numpy as np

from fpie import core_openmp
from fpie.io import read_images, write_image
from fpie.process import GridProcessor

# Must match the -C flag passed to likwid-perfctr
os.environ["OMP_NUM_THREADS"] = "8"


def main():
    # Initialize LIKWID marker API (must be before any markers)
    core_openmp.likwid_init()

    # 1. Load images
    print("Loading images...")
    try:
        src, mask, tgt = read_images(
            "test2_src.png", "test2_mask.png", "test2_target.png"
        )
    except Exception as e:
        print(f"Error loading images: {e}")
        print(
            "Please ensure test2_src.png, test2_mask.png, and test2_target.png exist."
        )
        return

    # 2. Initialize the OpenMP Grid Processor and swap the core for SolverV4
    print("Initializing OpenMP Solver v4...")
    proc = GridProcessor(
        gradient="src",
        backend="openmp",
        n_cpu=4,  # Adjust based on your hardware
        grid_x=8,
        grid_y=8,
    )
    # Swap the backend core for the new SolverV4
    proc.core = core_openmp.SolverV4(8, 8, 4)

    # 3. Reset the solver with input data
    # (0, 0) are the offsets for mask on src and (130, 130) for mask on tgt
    n_vars = proc.reset(src, mask, tgt, (0, 0), (260, 260))
    print(f"Number of variables to solve: {n_vars}")

    # 4. Run solver iterations
    print("Running solver...")
    start_time = time.time()

    # Run 5000 iterations in chunks of 100
    total_iters = 970
    chunk_size = 100
    result = tgt

    for i in range(0, total_iters, chunk_size):
        result, err = proc.step(chunk_size)
        print(f"Iteration {i + chunk_size}, Error: {err}")

    end_time = time.time()
    print(f"Solved in {end_time - start_time:.4f} seconds")

    # 5. Save the result
    write_image("result_v4.jpg", result)
    print("Saved result to result_v4.jpg")

    # Flush LIKWID marker data (must be before exit)
    core_openmp.likwid_close()


if __name__ == "__main__":
    main()
