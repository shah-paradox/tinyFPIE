# Fast Poisson Image Editing (FPIE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

FPIE is a high-performance implementation of the Poisson Image Editing algorithm, providing several backends for CPU and GPU acceleration. It supports seamless image cloning, texture flattening, and illumination compensation with a focus on speed and efficiency.

## 🚀 Key Features

- **Multiple Backends**: Support for `numpy`, `numba`, `taichi`, `gcc`, `openmp`, `mpi`, and `cuda`.
- **Advanced CPU Optimization**: 
  - **OpenMP**: Highly parallelized solvers for multi-core processors.
  - **AVX2 Vectorization**: Optimized kernels (V5) using manual AVX2 intrinsics for maximum throughput.
  - **Multigrid Solver**: Faster convergence using multigrid methods.
- **Hardware Profiling**: Integrated with **LIKWID** for granular performance monitoring (memory bandwidth, cache misses, etc.).
- **Flexible Interfaces**: Includes both a Command Line Interface (CLI) and a Graphical User Interface (GUI).
- **Scalable**: Benchmarking suite included to validate performance across different hardware configurations.

---

## 🛠️ System Setup

### Prerequisites

#### 1. System Dependencies
- **CMake** (3.5+)
- **GCC** (with OpenMP support)
- **LIKWID** (Optional, for performance profiling)
- **OpenCL / CUDA** (Optional, for GPU acceleration)

#### 2. Python Environment
- Python 3.10 to 3.13
- `numpy`, `opencv-python-headless`, `taichi`, `numba`

### Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/Trinkle23897/Fast-Poisson-Image-Editing.git
cd Fast-Poisson-Image-Editing
pip install -e .
```

This will automatically trigger the CMake build process for the C++ extensions (`fpie.core_openmp`, etc.).

---

## 📖 Usage

### Running Benchmarks

The repository is configured for high-performance benchmarking. Use the provided scripts to run different solver versions:

```bash
# Run the full benchmark suite with LIKWID profiling
python run_benchmarks.py

# Run a specific solver (e.g., V5 AVX2)
# Ensure config.py is set up correctly (or run run_benchmarks.py first)
python benchmark_v5.py
```

### Programmatic Usage

You can use the solvers in your own Python code:

```python
from fpie.process import GridProcessor
from fpie.io import read_images, write_image

# Load images
src, mask, tgt = read_images("src.png", "mask.png", "tgt.png")

# Initialize processor (e.g., OpenMP backend)
proc = GridProcessor(backend="openmp", n_cpu=8)

# Setup the solver
proc.reset(src, mask, tgt, (0, 0), (0, 0))

# Run iterations
result, error = proc.step(1000)

# Save result
write_image("output.png", result)
```

---

## 🏗️ Codebase Implementation Details

### Directory Structure

- `fpie/`: Core Python package.
  - `process.py`: Backend selection and processor abstractions (`EquProcessor`, `GridProcessor`).
  - `io.py`: Image loading and saving utilities.
  - `core/`: C++ source code.
    - `openmp/`: OpenMP and AVX2 optimized kernels.
      - `solver.cc`: PyBind11 bindings and solver interfaces.
      - `v5.cc`: High-performance AVX2 implementation.
      - `multigrid.cc`: Multigrid solver logic.
      - `equ.cc` / `grid.cc`: Standard Jacobi solvers.
- `pybind11/`: Submodule for Python/C++ interoperability.
- `tests/`: Dataset for validation and benchmarking.
- `run_benchmarks.py`: Automation script for performance profiling.

### Solver Architecture

FPIE implements two main processing strategies:
1. **EquProcessor**: Solves the Poisson equation only on the active pixels defined by the mask. It builds a sparse linear system.
2. **GridProcessor**: Solves the equation on a rectangular grid bounding the mask. This allows for efficient memory access and SIMD vectorization.

### Optimization Versions (OpenMP)
- **V1-V4**: Iterative optimizations for OpenMP parallelism and memory locality.
- **V5**: Hand-crafted AVX2 intrinsics to saturate CPU vector lanes, providing significant speedups on supported hardware.
- **Multigrid**: A hierarchical solver that reduces error across multiple scales, leading to much faster convergence than standard Jacobi methods.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a new branch for your feature.
3. Implement your changes (ensure C++ code follows the existing style).
4. Run benchmarks to ensure no performance regressions.
5. Submit a Pull Request.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
