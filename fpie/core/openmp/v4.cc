#include <omp.h>
#include <tuple>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include "likwid_wrapper.h"

#include "solver.h"

static thread_local bool likwid_v4_initialized = false;

OpenMPSolverV4::OpenMPSolverV4(int grid_x, int grid_y, int n_cpu)
    : imgbuf(NULL), r(NULL), p(NULL), Ap(NULL), GridSolver(grid_x, grid_y) {
  omp_set_num_threads(n_cpu);
}

OpenMPSolverV4::~OpenMPSolverV4() {
  if (imgbuf != NULL) delete[] imgbuf;
  if (r != NULL) delete[] r;
  if (p != NULL) delete[] p;
  if (Ap != NULL) delete[] Ap;
}

void OpenMPSolverV4::post_reset() {
  if (imgbuf != NULL) delete[] imgbuf;
  if (r != NULL) delete[] r;
  if (p != NULL) delete[] p;
  if (Ap != NULL) delete[] Ap;
  imgbuf = new unsigned char[N * M * 3];
  r = new float[N * M * 3]();
  p = new float[N * M * 3]();
  Ap = new float[N * M * 3]();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N * M * 3; i++) { r[i] = p[i] = Ap[i] = 0.0f; }
}

void OpenMPSolverV4::calc_error() {
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

std::tuple<py::array_t<unsigned char>, py::array_t<float>> OpenMPSolverV4::step(int iteration) {
  #pragma omp parallel
  {
    if (!likwid_v4_initialized) {
        LIKWID_MARKER_THREADINIT;
        likwid_v4_initialized = true;
    }
    LIKWID_MARKER_START("v4_compute");

    double r_sq_r = 0, r_sq_g = 0, r_sq_b = 0;
    #pragma omp for reduction(+:r_sq_r, r_sq_g, r_sq_b) schedule(static)
    for (int y = 1; y < N - 1; ++y) {
        for (int x = 1; x < M - 1; ++x) {
            int id = y * M + x;
            if (mask[id]) {
                int off3 = id * 3;
                int id0 = off3 - m3, id1 = off3 - 3, id2 = off3 + 3, id3 = off3 + m3;
                for (int c = 0; c < 3; ++c) {
                    r[off3+c] = grad[off3+c] + tgt[id0+c] + tgt[id1+c] + tgt[id2+c] + tgt[id3+c] - tgt[off3+c] * 4.0f;
                    p[off3+c] = r[off3+c];
                }
                r_sq_r += r[off3+0]*r[off3+0]; r_sq_g += r[off3+1]*r[off3+1]; r_sq_b += r[off3+2]*r[off3+2];
            }
        }
    }

    for (int it = 0; it < iteration; ++it) {
        double pAp_r = 0, pAp_g = 0, pAp_b = 0;
        #pragma omp for reduction(+:pAp_r, pAp_g, pAp_b) schedule(static)
        for (int y = 1; y < N - 1; ++y) {
            for (int x = 1; x < M - 1; ++x) {
                int id = y * M + x;
                if (mask[id]) {
                    int off3 = id * 3;
                    for (int c = 0; c < 3; ++c) {
                        float pt = mask[id-M] ? p[(id-M)*3+c] : 0.0f, pb = mask[id+M] ? p[(id+M)*3+c] : 0.0f, pl = mask[id-1] ? p[(id-1)*3+c] : 0.0f, pr = mask[id+1] ? p[(id+1)*3+c] : 0.0f;
                        Ap[off3+c] = 4.0f * p[off3+c] - (pt+pb+pl+pr);
                    }
                    pAp_r += p[off3+0]*Ap[off3+0]; pAp_g += p[off3+1]*Ap[off3+1]; pAp_b += p[off3+2]*Ap[off3+2];
                }
            }
        }
        float ar = (pAp_r > 1e-12) ? (r_sq_r / pAp_r) : 0, ag = (pAp_g > 1e-12) ? (r_sq_g / pAp_g) : 0, ab = (pAp_b > 1e-12) ? (r_sq_b / pAp_b) : 0;
        double r_new_r = 0, r_new_g = 0, r_new_b = 0;
        #pragma omp for reduction(+:r_new_r, r_new_g, r_new_b) schedule(static)
        for (int y = 1; y < N - 1; ++y) {
            for (int x = 1; x < M - 1; ++x) {
                int id = y * M + x;
                if (mask[id]) {
                    int off3 = id * 3;
                    for (int c = 0; c < 3; ++c) {
                        float aval = (c==0)?ar:(c==1)?ag:ab;
                        tgt[off3+c] += aval * p[off3+c]; r[off3+c] -= aval * Ap[off3+c];
                        if(c==0) r_new_r += r[off3+c]*r[off3+c]; else if(c==1) r_new_g += r[off3+c]*r[off3+c]; else r_new_b += r[off3+c]*r[off3+c];
                    }
                }
            }
        }
        float br = (r_sq_r > 1e-12) ? (r_new_r / r_sq_r) : 0, bg = (r_sq_g > 1e-12) ? (r_new_g / r_sq_g) : 0, bb = (r_sq_b > 1e-12) ? (r_new_b / r_sq_b) : 0;
        r_sq_r = r_new_r; r_sq_g = r_new_g; r_sq_b = r_new_b;
        #pragma omp for schedule(static)
        for (int y = 1; y < N - 1; ++y) {
            for (int x = 1; x < M - 1; ++x) {
                int id = y * M + x;
                if (mask[id]) {
                    int off3 = id * 3;
                    for (int c = 0; c < 3; ++c) {
                        float bval = (c==0)?br:(c==1)?bg:bb;
                        p[off3+c] = r[off3+c] + bval * p[off3+c];
                    }
                }
            }
        }
    }
    LIKWID_MARKER_STOP("v4_compute");
  }
  calc_error();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N * M * 3; ++i) imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : tgt[i];
  return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}
