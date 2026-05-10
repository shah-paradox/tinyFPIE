#include <omp.h>
#include <tuple>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include "likwid_wrapper.h"

#include "solver.h"

#define TILE_SIZE 64

static thread_local bool likwid_v3_initialized = false;

OpenMPSolverV3::OpenMPSolverV3(int grid_x, int grid_y, int n_cpu)
    : imgbuf(NULL), GridSolver(grid_x, grid_y) {
  omp_set_num_threads(n_cpu);
}

OpenMPSolverV3::~OpenMPSolverV3() {
  if (imgbuf != NULL) delete[] imgbuf;
}

void OpenMPSolverV3::post_reset() {
  if (imgbuf != NULL) delete[] imgbuf;
  imgbuf = new unsigned char[N * M * 3];
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N * M * 3; i++) { tgt[i] += 0.0f; grad[i] += 0.0f; }
}

void OpenMPSolverV3::calc_error() {
  double err_r = 0, err_g = 0, err_b = 0;
  #pragma omp parallel for reduction(+:err_r, err_g, err_b) schedule(static)
  for (int y = 1; y < N - 1; ++y) {
    for (int x = 1; x < M - 1; ++x) {
      int id = y * M + x;
      if (mask[id]) {
        int off3 = id * 3;
        int id0 = off3 - m3, id1 = off3 - 3, id2 = off3 + 3, id3 = off3 + m3;
        err_r += std::abs(grad[off3+0] + tgt[id0+0] + tgt[id1+0] + tgt[id2+0] + tgt[id3+0] - tgt[off3+0]*4.0f);
        err_g += std::abs(grad[off3+1] + tgt[id0+1] + tgt[id1+1] + tgt[id2+1] + tgt[id3+1] - tgt[off3+1]*4.0f);
        err_b += std::abs(grad[off3+2] + tgt[id0+2] + tgt[id1+2] + tgt[id2+2] + tgt[id3+2] - tgt[off3+2]*4.0f);
      }
    }
  }
  err[0] = (float)err_r; err[1] = (float)err_g; err[2] = (float)err_b;
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>> OpenMPSolverV3::step(int iteration) {
  #pragma omp parallel
  {
    if (!likwid_v3_initialized) {
        LIKWID_MARKER_THREADINIT;
        likwid_v3_initialized = true;
    }
    LIKWID_MARKER_START("v3_compute");

    for (int it = 0; it < iteration; ++it) {
        // --- RED SWEEP ---
        #pragma omp for collapse(2) schedule(static)
        for (int ti = 1; ti < N - 1; ti += TILE_SIZE) {
            for (int tj = 1; tj < M - 1; tj += TILE_SIZE) {
                int i_end = std::min(ti + TILE_SIZE, N - 1);
                int j_end = std::min(tj + TILE_SIZE, M - 1);
                for (int i = ti; i < i_end; i++) {
                    int j_start = tj + ((i + tj) % 2 == 0 ? 0 : 1);
                    #pragma omp simd
                    for (int j = j_start; j < j_end; j += 2) {
                        int id = i * M + j;
                        if (mask[id]) {
                            int off3 = id * 3;
                            int id0 = off3 - m3, id1 = off3 - 3, id2 = off3 + 3, id3 = off3 + m3;
                            tgt[off3+0] = 0.25f * (tgt[id0+0] + tgt[id1+0] + tgt[id2+0] + tgt[id3+0] + grad[off3+0]);
                            tgt[off3+1] = 0.25f * (tgt[id0+1] + tgt[id1+1] + tgt[id2+1] + tgt[id3+1] + grad[off3+1]);
                            tgt[off3+2] = 0.25f * (tgt[id0+2] + tgt[id1+2] + tgt[id2+2] + tgt[id3+2] + grad[off3+2]);
                        }
                    }
                }
            }
        }
        // --- BLACK SWEEP ---
        #pragma omp for collapse(2) schedule(static)
        for (int ti = 1; ti < N - 1; ti += TILE_SIZE) {
            for (int tj = 1; tj < M - 1; tj += TILE_SIZE) {
                int i_end = std::min(ti + TILE_SIZE, N - 1);
                int j_end = std::min(tj + TILE_SIZE, M - 1);
                for (int i = ti; i < i_end; i++) {
                    int j_start = tj + ((i + tj) % 2 == 1 ? 0 : 1);
                    #pragma omp simd
                    for (int j = j_start; j < j_end; j += 2) {
                        int id = i * M + j;
                        if (mask[id]) {
                            int off3 = id * 3;
                            int id0 = off3 - m3, id1 = off3 - 3, id2 = off3 + 3, id3 = off3 + m3;
                            tgt[off3+0] = 0.25f * (tgt[id0+0] + tgt[id1+0] + tgt[id2+0] + tgt[id3+0] + grad[off3+0]);
                            tgt[off3+1] = 0.25f * (tgt[id0+1] + tgt[id1+1] + tgt[id2+1] + tgt[id3+1] + grad[off3+1]);
                            tgt[off3+2] = 0.25f * (tgt[id0+2] + tgt[id1+2] + tgt[id2+2] + tgt[id3+2] + grad[off3+2]);
                        }
                    }
                }
            }
        }
    }
    LIKWID_MARKER_STOP("v3_compute");
  }
  calc_error();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N * M * 3; ++i) imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : tgt[i];
  return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}
