#include <omp.h>
#include <tuple>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "likwid_wrapper.h"

#include "solver.h"

static thread_local bool likwid_multigrid_v2_initialized = false;

OpenMPMultigridSolver::OpenMPMultigridSolver(int grid_x, int grid_y, int n_cpu)
    : imgbuf(NULL), tmp(NULL), mask_c(NULL), tgt_c(NULL), grad_c(NULL), GridSolver(grid_x, grid_y) {
  omp_set_num_threads(n_cpu);
}

OpenMPMultigridSolver::~OpenMPMultigridSolver() {
  if (imgbuf != NULL) {
    delete[] imgbuf;
    delete[] mask_c;
    delete[] tgt_c;
    delete[] grad_c;
  }
}

void OpenMPMultigridSolver::post_reset() {
  if (imgbuf != NULL) {
    delete[] imgbuf;
    delete[] mask_c;
    delete[] tgt_c;
    delete[] grad_c;
  }

  imgbuf = new unsigned char[N * M * 3];

  N_c = N / 2;
  M_c = M / 2;
  m3_c = M_c * 3;

  mask_c = new int[N_c * M_c]();
  tgt_c = new float[N_c * m3_c]();
  grad_c = new float[N_c * m3_c]();
}

inline void OpenMPMultigridSolver::smooth_parallel(int n_rows, int n_cols, float* t, const float* g, const int* msk, int m3_val) {
    const float INV_4 = 0.25f;
    for (int color_pass = 0; color_pass < 2; ++color_pass) {
        #pragma omp for schedule(static)
        for (int y = 1; y < n_rows - 1; ++y) {
            int m_off = y * n_cols;
            int off3_base = m_off * 3;
            int y_parity = y & 1;
            #pragma omp simd
            for (int x = 1; x < n_cols - 1; ++x) {
                int is_active_color = ((y_parity ^ (x & 1)) == color_pass);
                int id = m_off + x;
                if (msk[id] && is_active_color) {
                    int off3 = off3_base + x * 3;
                    int id0 = off3 - m3_val;
                    int id1 = off3 - 3;
                    int id2 = off3 + 3;
                    int id3 = off3 + m3_val;
                    t[off3 + 0] = (g[off3 + 0] + t[id0 + 0] + t[id1 + 0] + t[id2 + 0] + t[id3 + 0]) * INV_4;
                    t[off3 + 1] = (g[off3 + 1] + t[id0 + 1] + t[id1 + 1] + t[id2 + 1] + t[id3 + 1]) * INV_4;
                    t[off3 + 2] = (g[off3 + 2] + t[id0 + 2] + t[id1 + 2] + t[id2 + 2] + t[id3 + 2]) * INV_4;
                }
            }
        }
    }
}

void OpenMPMultigridSolver::restrict_residual_parallel() {
  #pragma omp for schedule(static)
  for (int i = 1; i < N_c - 1; ++i) {
    for (int j = 1; j < M_c - 1; ++j) {
      int id_c = i * M_c + j;
      int off3_c = id_c * 3;
      int i_f = i * 2;
      int j_f = j * 2;
      int id_f = i_f * M + j_f;
      int off3_f = id_f * 3;
      mask_c[id_c] = mask[id_f] | mask[id_f + 1] | mask[id_f + M] | mask[id_f + M + 1];
      if (mask_c[id_c]) {
        float t_val, Ax, res;
        t_val = tgt[off3_f + 0];
        Ax = 4.0f * t_val - tgt[off3_f - m3 + 0] - tgt[off3_f + m3 + 0] - tgt[off3_f - 3 + 0] - tgt[off3_f + 3 + 0];
        res = grad[off3_f + 0] - Ax;
        grad_c[off3_c + 0] = res * 4.0f;
        tgt_c[off3_c + 0] = 0.0f;
        t_val = tgt[off3_f + 1];
        Ax = 4.0f * t_val - tgt[off3_f - m3 + 1] - tgt[off3_f + m3 + 1] - tgt[off3_f - 3 + 1] - tgt[off3_f + 3 + 1];
        res = grad[off3_f + 1] - Ax;
        grad_c[off3_c + 1] = res * 4.0f;
        tgt_c[off3_c + 1] = 0.0f;
        t_val = tgt[off3_f + 2];
        Ax = 4.0f * t_val - tgt[off3_f - m3 + 2] - tgt[off3_f + m3 + 2] - tgt[off3_f - 3 + 2] - tgt[off3_f + 3 + 2];
        res = grad[off3_f + 2] - Ax;
        grad_c[off3_c + 2] = res * 4.0f;
        tgt_c[off3_c + 2] = 0.0f;
      } else {
        tgt_c[off3_c + 0] = tgt_c[off3_c + 1] = tgt_c[off3_c + 2] = 0.0f;
      }
    }
  }
}

void OpenMPMultigridSolver::prolongate_correction_parallel() {
  #pragma omp for schedule(static)
  for (int i = 1; i < N_c - 1; ++i) {
    for (int j = 1; j < M_c - 1; ++j) {
      int id_c = i * M_c + j;
      if (!mask_c[id_c]) continue;
      int off3_c = id_c * 3;
      float tc0 = tgt_c[off3_c + 0];
      float tc1 = tgt_c[off3_c + 1];
      float tc2 = tgt_c[off3_c + 2];
      int i_f = i * 2, j_f = j * 2, id_f, off3_f;
      id_f = i_f * M + j_f;
      if (mask[id_f]) { off3_f = id_f * 3; tgt[off3_f+0] += tc0; tgt[off3_f+1] += tc1; tgt[off3_f+2] += tc2; }
      id_f = i_f * M + (j_f + 1);
      if (mask[id_f]) { off3_f = id_f * 3; tgt[off3_f+0] += tc0; tgt[off3_f+1] += tc1; tgt[off3_f+2] += tc2; }
      id_f = (i_f + 1) * M + j_f;
      if (mask[id_f]) { off3_f = id_f * 3; tgt[off3_f+0] += tc0; tgt[off3_f+1] += tc1; tgt[off3_f+2] += tc2; }
      id_f = (i_f + 1) * M + (j_f + 1);
      if (mask[id_f]) { off3_f = id_f * 3; tgt[off3_f+0] += tc0; tgt[off3_f+1] += tc1; tgt[off3_f+2] += tc2; }
    }
  }
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>> OpenMPMultigridSolver::step(int iteration) {
  #pragma omp parallel
  {
    if (!likwid_multigrid_v2_initialized) {
        LIKWID_MARKER_THREADINIT;
        likwid_multigrid_v2_initialized = true;
    }
    LIKWID_MARKER_START("multigrid");

    for (int it = 0; it < iteration; ++it) {
      smooth_parallel(N, M, tgt, grad, mask, m3);
      restrict_residual_parallel();
      for (int c_it = 0; c_it < 2; ++c_it) {
        smooth_parallel(N_c, M_c, tgt_c, grad_c, mask_c, m3_c);
      }
      prolongate_correction_parallel();
      smooth_parallel(N, M, tgt, grad, mask, m3);
    }
    LIKWID_MARKER_STOP("multigrid");
  }

  // Error and output copy can stay as separate parallel regions or be merged
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

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N * M * 3; ++i) {
    imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : tgt[i];
  }

  return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}
