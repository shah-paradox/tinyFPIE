#include <omp.h>
#include <tuple>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include "likwid_wrapper.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "solver.h"

static thread_local bool likwid_v5_initialized = false;

OpenMPSolverV5::OpenMPSolverV5(int grid_x, int grid_y, int n_cpu)
    : imgbuf(NULL), r(NULL), p(NULL), Ap(NULL), GridSolver(grid_x, grid_y) {
  omp_set_num_threads(n_cpu);
}

OpenMPSolverV5::~OpenMPSolverV5() {
  if (imgbuf != NULL) delete[] imgbuf;
  if (r != NULL) delete[] r;
  if (p != NULL) delete[] p;
  if (Ap != NULL) delete[] Ap;
}

void OpenMPSolverV5::post_reset() {
  if (imgbuf != NULL) delete[] imgbuf;
  if (r != NULL) delete[] r;
  if (p != NULL) delete[] p;
  if (Ap != NULL) delete[] Ap;
  
  imgbuf = new unsigned char[N * M * 3];
  r = new float[N * M * 3]();
  p = new float[N * M * 3]();
  Ap = new float[N * M * 3]();
  
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
          int off3 = (i * M + j) * 3;
          tgt[off3 + 0] += 0.0f;
          grad[off3 + 0] += 0.0f;
          r[off3 + 0] = 0.0f;
          p[off3 + 0] = 0.0f;
          Ap[off3 + 0] = 0.0f;
      }
  }
}

#ifdef __AVX2__
static inline __m256 gather_channel(const float* base_ptr, __m256i idx_vec) {
    return _mm256_i32gather_ps(base_ptr, idx_vec, 1);
}
static inline void scatter_channel(float* base_ptr, __m256i idx_vec, __m256 vals) {
    _mm256_i32scatter_ps(base_ptr, idx_vec, vals, 1);
}
static inline __m256i make_offset_vec(int y, int x, int c, int M) {
    int base = (y * M + x);
    alignas(32) int off[8];
    for (int k = 0; k < 8; ++k) off[k] = ((base + k) * 3 + c) * sizeof(float);
    return _mm256_load_si256((__m256i*)off);
}
static inline __m256i load_mask(const unsigned char* mask_base, int y, int x, int M) {
    int base = y * M + x;
    __m128i mask8 = _mm_loadl_epi64((__m128i const*)(mask_base + base));
    __m256i mask32 = _mm256_cvtepu8_epi32(mask8);
    return _mm256_cmpgt_epi32(mask32, _mm256_setzero_si256());
}
#endif

void OpenMPSolverV5::calc_error() {
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

std::tuple<py::array_t<unsigned char>, py::array_t<float>> OpenMPSolverV5::step(int max_iterations) {
  #pragma omp parallel
  {
    if (!likwid_v5_initialized) {
        LIKWID_MARKER_THREADINIT;
        likwid_v5_initialized = true;
    }
    LIKWID_MARKER_START("v5_compute");

    double r_sq_r = 0, r_sq_g = 0, r_sq_b = 0;
#ifdef __AVX2__
    #pragma omp for reduction(+:r_sq_r, r_sq_g, r_sq_b) schedule(static)
    for (int y = 1; y < N - 1; ++y) {
        double lr = 0, lg = 0, lb = 0;
        for (int x = 1; x <= M - 2; x += 8) {
            if (x + 7 > M - 2) break;
            for (int c = 0; c < 3; ++c) {
                __m256i off_c = make_offset_vec(y, x, c, M);
                __m256i m32 = load_mask(mask, y, x, M);
                __m256 mps = _mm256_castsi256_ps(m32);
                __m256 g = gather_channel(grad, off_c);
                __m256 tc = gather_channel(tgt, off_c), tt = gather_channel(tgt, make_offset_vec(y-1,x,c,M)), tb = gather_channel(tgt, make_offset_vec(y+1,x,c,M)), tl = gather_channel(tgt, make_offset_vec(y,x-1,c,M)), tr = gather_channel(tgt, make_offset_vec(y,x+1,c,M));
                __m256 res = _mm256_add_ps(g, _mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(tt, tb), _mm256_add_ps(tl, tr)), _mm256_mul_ps(tc, _mm256_set1_ps(4.0f))));
                res = _mm256_and_ps(res, mps);
                scatter_channel(r, off_c, res); scatter_channel(p, off_c, res);
                __m256 r2 = _mm256_mul_ps(res, res);
                float tmp[8]; _mm256_storeu_ps(tmp, r2);
                double s = 0; for (int k = 0; k < 8; ++k) s += tmp[k];
                if(c==0) lr+=s; else if(c==1) lg+=s; else lb+=s;
            }
        }
        r_sq_r += lr; r_sq_g += lg; r_sq_b += lb;
    }
#else
    #pragma omp for reduction(+:r_sq_r, r_sq_g, r_sq_b) schedule(static)
    for (int y = 1; y < N - 1; ++y) {
        for (int x = 1; x < M - 1; ++x) {
            int id = y * M + x;
            if (mask[id]) {
                int off3 = id * 3;
                for (int c = 0; c < 3; ++c) {
                    r[off3+c] = grad[off3+c] + tgt[(id-M)*3+c] + tgt[(id+M)*3+c] + tgt[(id-1)*3+c] + tgt[(id+1)*3+c] - tgt[off3+c] * 4.0f;
                    p[off3+c] = r[off3+c];
                }
                r_sq_r += r[off3+0]*r[off3+0]; r_sq_g += r[off3+1]*r[off3+1]; r_sq_b += r[off3+2]*r[off3+2];
            }
        }
    }
#endif

    double tolerance = 15.0;
    int it = 0;
    while (it < max_iterations) {
        double pAp_r = 0, pAp_g = 0, pAp_b = 0;
#ifdef __AVX2__
        #pragma omp for reduction(+:pAp_r, pAp_g, pAp_b) schedule(static)
        for (int y = 1; y < N - 1; ++y) {
            double lr = 0, lg = 0, lb = 0;
            for (int x = 1; x <= M - 2; x += 8) {
                if (x + 7 > M - 2) break;
                for (int c = 0; c < 3; ++c) {
                    __m256i off_c = make_offset_vec(y, x, c, M);
                    __m256i m32 = load_mask(mask, y, x, M);
                    __m256 mps = _mm256_castsi256_ps(m32);
                    __m256 pc = gather_channel(p, off_c), pt = gather_channel(p, make_offset_vec(y-1,x,c,M)), pb = gather_channel(p, make_offset_vec(y+1,x,c,M)), pl = gather_channel(p, make_offset_vec(y,x-1,c,M)), pr = gather_channel(p, make_offset_vec(y,x+1,c,M));
                    pt = _mm256_and_ps(pt, _mm256_castsi256_ps(load_mask(mask, y-1, x, M)));
                    pb = _mm256_and_ps(pb, _mm256_castsi256_ps(load_mask(mask, y+1, x, M)));
                    pl = _mm256_and_ps(pl, _mm256_castsi256_ps(load_mask(mask, y, x-1, M)));
                    pr = _mm256_and_ps(pr, _mm256_castsi256_ps(load_mask(mask, y, x+1, M)));
                    __m256 ap = _mm256_and_ps(_mm256_sub_ps(_mm256_mul_ps(pc, _mm256_set1_ps(4.0f)), _mm256_add_ps(_mm256_add_ps(pt, pb), _mm256_add_ps(pl, pr))), mps);
                    scatter_channel(Ap, off_c, ap);
                    __m256 prod = _mm256_mul_ps(pc, ap);
                    float tmp[8]; _mm256_storeu_ps(tmp, prod);
                    double s = 0; for (int k = 0; k < 8; ++k) s += tmp[k];
                    if(c==0) lr+=s; else if(c==1) lg+=s; else lb+=s;
                }
            }
            pAp_r += lr; pAp_g += lg; pAp_b += lb;
        }
#else
        #pragma omp for reduction(+:pAp_r, pAp_g, pAp_b) schedule(static)
        for (int y = 1; y < N - 1; ++y) {
            for (int x = 1; x < M - 1; ++x) {
                int id = y * M + x;
                if (mask[id]) {
                    int off3 = id * 3;
                    for (int c = 0; c < 3; ++c) {
                        float pt = mask[id-M] ? p[(id-M)*3+c] : 0.0f, pb = mask[id+M] ? p[(id+M)*3+c] : 0.0f, pl = mask[id-1] ? p[(id-1)*3+c] : 0.0f, pr = mask[id+1] ? p[(id+1)*3+c] : 0.0f;
                        Ap[off3+c] = 4.0f * p[off3+c] - (pt+pb+pl+pr);
                        if(c==0) pAp_r += p[off3+c]*Ap[off3+c]; else if(c==1) pAp_g += p[off3+c]*Ap[off3+c]; else pAp_b += p[off3+c]*Ap[off3+c];
                    }
                }
            }
        }
#endif
        float ar = (pAp_r > 1e-12) ? (r_sq_r / pAp_r) : 0, ag = (pAp_g > 1e-12) ? (r_sq_g / pAp_g) : 0, ab = (pAp_b > 1e-12) ? (r_sq_b / pAp_b) : 0;
        double r_new_r = 0, r_new_g = 0, r_new_b = 0;
#ifdef __AVX2__
        #pragma omp for reduction(+:r_new_r, r_new_g, r_new_b) schedule(static)
        for (int y = 1; y < N - 1; ++y) {
            double lr = 0, lg = 0, lb = 0;
            for (int x = 1; x <= M - 2; x += 8) {
                if (x + 7 > M - 2) break;
                for (int c = 0; c < 3; ++c) {
                    __m256i off = make_offset_vec(y, x, c, M), m32 = load_mask(mask, y, x, M); __m256 mps = _mm256_castsi256_ps(m32);
                    __m256 t = gather_channel(tgt, off), pc = gather_channel(p, off), ap = gather_channel(Ap, off);
                    float aval = (c==0)?ar:(c==1)?ag:ab; __m256 a = _mm256_set1_ps(aval);
                    scatter_channel(tgt, off, _mm256_blendv_ps(t, _mm256_fmadd_ps(a, pc, t), mps));
                    __m256 nr = _mm256_and_ps(_mm256_fnmadd_ps(a, ap, gather_channel(r, off)), mps);
                    scatter_channel(r, off, nr);
                    __m256 nr2 = _mm256_mul_ps(nr, nr); float tmp[8]; _mm256_storeu_ps(tmp, nr2);
                    double s = 0; for(int k=0; k<8; ++k) s+=tmp[k];
                    if(c==0) lr+=s; else if(c==1) lg+=s; else lb+=s;
                }
            }
            r_new_r += lr; r_new_g += lg; r_new_b += lb;
        }
#else
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
#endif
        float br = (r_sq_r > 1e-12) ? (r_new_r / r_sq_r) : 0, bg = (r_sq_g > 1e-12) ? (r_new_g / r_sq_g) : 0, bb = (r_sq_b > 1e-12) ? (r_new_b / r_sq_b) : 0;
        r_sq_r = r_new_r; r_sq_g = r_new_g; r_sq_b = r_new_b;
#ifdef __AVX2__
        #pragma omp for schedule(static)
        for (int y = 1; y < N - 1; ++y) {
            for (int x = 1; x <= M - 2; x += 8) {
                if (x + 7 > M - 2) break;
                for (int c = 0; c < 3; ++c) {
                    __m256i off = make_offset_vec(y, x, c, M), m32 = load_mask(mask, y, x, M); __m256 mps = _mm256_castsi256_ps(m32);
                    float bval = (c==0)?br:(c==1)?bg:bb; __m256 b = _mm256_set1_ps(bval);
                    scatter_channel(p, off, _mm256_and_ps(_mm256_fmadd_ps(b, gather_channel(p, off), gather_channel(r, off)), mps));
                }
            }
        }
#else
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
#endif
        it++;
        if (it % 100 == 0) {
            // Simplified error check
            double current_err = std::sqrt(r_sq_r + r_sq_g + r_sq_b) / (N*M);
            if (current_err < tolerance/1000.0) break; 
        }
    }
    LIKWID_MARKER_STOP("v5_compute");
  }
  calc_error();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N * M * 3; ++i) imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : tgt[i];
  return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}
