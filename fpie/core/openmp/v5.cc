#include <omp.h>
#include <tuple>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include "likwid_wrapper.h"

// AVX2 intrinsics
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
  
  // NUMA First-Touch Allocation
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

// ---------------------------------------------------------------------------
// AVX2 helper functions
// ---------------------------------------------------------------------------
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
    for (int k = 0; k < 8; ++k)
        off[k] = ((base + k) * 3 + c) * sizeof(float);
    return _mm256_load_si256((__m256i*)off);
}

// FIX 1: Mask reversal fixed here
static inline __m256i load_mask(const unsigned char* mask_base, int y, int x, int M) {
    int base = y * M + x;
    __m128i mask8 = _mm_loadl_epi64((__m128i const*)(mask_base + base));
    __m256i mask32 = _mm256_cvtepu8_epi32(mask8);
    // Use _mm256_cmpgt_epi32 to check if mask > 0 (Active pixel)
    return _mm256_cmpgt_epi32(mask32, _mm256_setzero_si256());
}

#endif // __AVX2__

// ---------------------------------------------------------------------------
// Core Solver
// ---------------------------------------------------------------------------

void OpenMPSolverV5::calc_error() {
  double err_r = 0, err_g = 0, err_b = 0;
  
  #pragma omp parallel for reduction(+:err_r, err_g, err_b) schedule(static)
  for (int y = 1; y < N - 1; ++y) {
    for (int x = 1; x < M - 1; ++x) {
      int id = y * M + x;
      if (mask[id]) {
        int off3 = id * 3;
        int id0 = off3 - m3;
        int id1 = off3 - 3;
        int id2 = off3 + 3;
        int id3 = off3 + m3;
        
        err_r += std::abs(grad[off3 + 0] + tgt[id0 + 0] + tgt[id1 + 0] + tgt[id2 + 0] + tgt[id3 + 0] - tgt[off3 + 0] * 4.0f);
        err_g += std::abs(grad[off3 + 1] + tgt[id0 + 1] + tgt[id1 + 1] + tgt[id2 + 1] + tgt[id3 + 1] - tgt[off3 + 1] * 4.0f);
        err_b += std::abs(grad[off3 + 2] + tgt[id0 + 2] + tgt[id1 + 2] + tgt[id2 + 2] + tgt[id3 + 2] - tgt[off3 + 2] * 4.0f);
      }
    }
  }
  err[0] = (float)err_r; 
  err[1] = (float)err_g; 
  err[2] = (float)err_b;
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>> OpenMPSolverV5::step(int max_iterations) {
  #pragma omp parallel
  {
      if (!likwid_v5_initialized) {
          LIKWID_MARKER_THREADINIT;
          likwid_v5_initialized = true;
      }
  }

  LIKWID_MARKER_START("v5_compute");

  double r_sq_r = 0, r_sq_g = 0, r_sq_b = 0;

  // =========================================================================
  // STEP 0: INITIALIZE RESIDUALS (Run ONCE)
  // =========================================================================
#ifdef __AVX2__
  #pragma omp parallel for reduction(+:r_sq_r, r_sq_g, r_sq_b) schedule(static)
  for (int y = 1; y < N - 1; ++y) {
      double local_r_sq_r = 0, local_r_sq_g = 0, local_r_sq_b = 0;
      for (int x = 1; x <= M - 2; x += 8) {
          int xend = (x + 7 <= M - 2) ? 8 : (M - 1 - x);
          if (xend < 8) break; 

          for (int c = 0; c < 3; ++c) {
              __m256i off_c  = make_offset_vec(y, x, c, M);
              __m256i off_t  = make_offset_vec(y-1, x, c, M);
              __m256i off_b  = make_offset_vec(y+1, x, c, M);
              __m256i off_l  = make_offset_vec(y, x-1, c, M);
              __m256i off_r  = make_offset_vec(y, x+1, c, M);

              __m256i mask32 = load_mask(mask, y, x, M);
              __m256 mask_ps = _mm256_castsi256_ps(mask32);

              __m256 g = gather_channel(grad, off_c);
              __m256 t_c = gather_channel(tgt, off_c);
              __m256 t_t = gather_channel(tgt, off_t);
              __m256 t_b = gather_channel(tgt, off_b);
              __m256 t_l = gather_channel(tgt, off_l);
              __m256 t_r = gather_channel(tgt, off_r);

              __m256 nb_sum = _mm256_add_ps(_mm256_add_ps(t_t, t_b), _mm256_add_ps(t_l, t_r));
              __m256 res = _mm256_add_ps(g, _mm256_sub_ps(nb_sum, _mm256_mul_ps(t_c, _mm256_set1_ps(4.0f))));
              res = _mm256_and_ps(res, mask_ps);

              scatter_channel(r, off_c, res);
              scatter_channel(p, off_c, res);

              __m256 r2 = _mm256_mul_ps(res, res);
              float tmp[8];
              _mm256_storeu_ps(tmp, r2);
              double sum = 0;
              for (int k = 0; k < 8; ++k) sum += tmp[k];
              
              if(c == 0) local_r_sq_r += sum;
              if(c == 1) local_r_sq_g += sum;
              if(c == 2) local_r_sq_b += sum;
          } 
      } 

      for (int x = ((M-2) & ~7) + 1; x <= M-2; ++x) {
          int id = y * M + x;
          if (mask[id]) {
              int off3 = id * 3;
              for (int c = 0; c < 3; ++c) {
                  float r_val = grad[off3+c] + tgt[(id-M)*3+c] + tgt[(id+M)*3+c]
                               + tgt[(id-1)*3+c] + tgt[(id+1)*3+c] - 4.0f * tgt[off3+c];
                  r[off3+c]  = r_val;
                  p[off3+c]  = r_val;
                  if(c==0) local_r_sq_r += r_val * r_val;
                  if(c==1) local_r_sq_g += r_val * r_val;
                  if(c==2) local_r_sq_b += r_val * r_val;
              }
          }
      }
      r_sq_r += local_r_sq_r;
      r_sq_g += local_r_sq_g;
      r_sq_b += local_r_sq_b;
  }
#else
  #pragma omp parallel for reduction(+:r_sq_r, r_sq_g, r_sq_b) schedule(static)
  for (int y = 1; y < N - 1; ++y) {
    for (int x = 1; x < M - 1; ++x) {
      int id = y * M + x;
      if (mask[id]) {
        int off3 = id * 3;
        int id0 = off3 - m3;
        int id1 = off3 - 3;
        int id2 = off3 + 3;
        int id3 = off3 + m3;
        for (int c = 0; c < 3; ++c) {
          r[off3+c] = grad[off3+c] + tgt[id0+c] + tgt[id1+c] + tgt[id2+c] + tgt[id3+c] - tgt[off3+c] * 4.0f;
          p[off3+c] = r[off3+c];
        }
        r_sq_r += r[off3+0]*r[off3+0];
        r_sq_g += r[off3+1]*r[off3+1];
        r_sq_b += r[off3+2]*r[off3+2];
      }
    }
  }
#endif

  // =========================================================================
  // FIX 3: CONTINUOUS LOOP WITH TOLERANCE CHECK
  // =========================================================================
  double tolerance = 15.0; // The threshold for visual perfection
  double current_err = 999999.0;
  int it = 0;

  while (current_err > tolerance && it < max_iterations) {
      
      double pAp_r = 0, pAp_g = 0, pAp_b = 0;

#ifdef __AVX2__
      // 1. Ap = A * p  and pAp
      #pragma omp parallel for reduction(+:pAp_r, pAp_g, pAp_b) schedule(static)
      for (int y = 1; y < N - 1; ++y) {
          double local_pAp_r = 0, local_pAp_g = 0, local_pAp_b = 0;
          for (int x = 1; x <= M - 2; x += 8) {
              int xend = (x + 7 <= M - 2) ? 8 : (M - 1 - x);
              if (xend < 8) break;

              for (int c = 0; c < 3; ++c) {
                  __m256i off_c = make_offset_vec(y, x, c, M);
                  __m256i off_t = make_offset_vec(y-1, x, c, M);
                  __m256i off_b = make_offset_vec(y+1, x, c, M);
                  __m256i off_l = make_offset_vec(y, x-1, c, M);
                  __m256i off_r = make_offset_vec(y, x+1, c, M);

                  __m256i mask32 = load_mask(mask, y, x, M);
                  __m256 mask_ps = _mm256_castsi256_ps(mask32);

                  __m256 p_c = gather_channel(p, off_c);
                  __m256 p_t = gather_channel(p, off_t);
                  __m256 p_b = gather_channel(p, off_b);
                  __m256 p_l = gather_channel(p, off_l);
                  __m256 p_r = gather_channel(p, off_r);

                  __m256i mask_t = load_mask(mask, y-1, x, M);
                  __m256i mask_b = load_mask(mask, y+1, x, M);
                  __m256i mask_l = load_mask(mask, y, x-1, M);
                  __m256i mask_r = load_mask(mask, y, x+1, M);
                  
                  p_t = _mm256_and_ps(p_t, _mm256_castsi256_ps(mask_t));
                  p_b = _mm256_and_ps(p_b, _mm256_castsi256_ps(mask_b));
                  p_l = _mm256_and_ps(p_l, _mm256_castsi256_ps(mask_l));
                  p_r = _mm256_and_ps(p_r, _mm256_castsi256_ps(mask_r));

                  __m256 nb = _mm256_add_ps(_mm256_add_ps(p_t, p_b), _mm256_add_ps(p_l, p_r));
                  __m256 Ap_val = _mm256_sub_ps(_mm256_mul_ps(p_c, _mm256_set1_ps(4.0f)), nb);
                  Ap_val = _mm256_and_ps(Ap_val, mask_ps);

                  scatter_channel(Ap, off_c, Ap_val);

                  __m256 prod = _mm256_mul_ps(p_c, Ap_val);
                  float tmp[8];
                  _mm256_storeu_ps(tmp, prod);
                  double sum = 0;
                  for (int k = 0; k < 8; ++k) sum += tmp[k];
                  
                  if(c == 0) local_pAp_r += sum;
                  if(c == 1) local_pAp_g += sum;
                  if(c == 2) local_pAp_b += sum;
              }
          }
          
          for (int x = ((M-2) & ~7) + 1; x <= M-2; ++x) {
              int id = y * M + x;
              if (mask[id]) {
                  int off3 = id * 3;
                  for (int c = 0; c < 3; ++c) {
                      float p_t = mask[id-M] ? p[(id-M)*3+c] : 0.0f;
                      float p_b = mask[id+M] ? p[(id+M)*3+c] : 0.0f;
                      float p_l = mask[id-1] ? p[(id-1)*3+c] : 0.0f;
                      float p_r = mask[id+1] ? p[(id+1)*3+c] : 0.0f;
                      float Ap_val = 4.0f * p[off3+c] - (p_t+p_b+p_l+p_r);
                      Ap[off3+c] = Ap_val;
                      
                      if(c==0) local_pAp_r += p[off3+c] * Ap_val;
                      if(c==1) local_pAp_g += p[off3+c] * Ap_val;
                      if(c==2) local_pAp_b += p[off3+c] * Ap_val;
                  }
              }
          }
          pAp_r += local_pAp_r;
          pAp_g += local_pAp_g;
          pAp_b += local_pAp_b;
      }
#else
      #pragma omp parallel for reduction(+:pAp_r, pAp_g, pAp_b) schedule(static)
      for (int y = 1; y < N - 1; ++y) {
        for (int x = 1; x < M - 1; ++x) {
          int id = y * M + x;
          if (mask[id]) {
            int off3 = id * 3;
            for (int c = 0; c < 3; ++c) {
              float p_top = mask[id-M] ? p[(id-M)*3+c] : 0.0f;
              float p_bot = mask[id+M] ? p[(id+M)*3+c] : 0.0f;
              float p_lft = mask[id-1] ? p[(id-1)*3+c] : 0.0f;
              float p_rgt = mask[id+1] ? p[(id+1)*3+c] : 0.0f;
              Ap[off3+c] = 4.0f * p[off3+c] - (p_top+p_bot+p_lft+p_rgt);
            }
            pAp_r += p[off3+0] * Ap[off3+0];
            pAp_g += p[off3+1] * Ap[off3+1];
            pAp_b += p[off3+2] * Ap[off3+2];
          }
        }
      }
#endif

      float alpha_r = (pAp_r > 1e-12) ? (r_sq_r / pAp_r) : 0.0f;
      float alpha_g = (pAp_g > 1e-12) ? (r_sq_g / pAp_g) : 0.0f;
      float alpha_b = (pAp_b > 1e-12) ? (r_sq_b / pAp_b) : 0.0f;

      double r_sq_new_r = 0, r_sq_new_g = 0, r_sq_new_b = 0;

#ifdef __AVX2__
      // 2. Update tgt and r
      #pragma omp parallel for reduction(+:r_sq_new_r, r_sq_new_g, r_sq_new_b) schedule(static)
      for (int y = 1; y < N - 1; ++y) {
          double local_new_r = 0, local_new_g = 0, local_new_b = 0;
          for (int x = 1; x <= M - 2; x += 8) {
              int xend = (x + 7 <= M - 2) ? 8 : (M - 1 - x);
              if (xend < 8) break;

              for (int c = 0; c < 3; ++c) {
                  __m256i off_c = make_offset_vec(y, x, c, M);
                  __m256i mask32 = load_mask(mask, y, x, M);
                  __m256 mask_ps = _mm256_castsi256_ps(mask32);

                  __m256 t = gather_channel(tgt, off_c);
                  __m256 p_c = gather_channel(p, off_c);
                  __m256 ap = gather_channel(Ap, off_c);

                  float a_val = (c == 0) ? alpha_r : ((c == 1) ? alpha_g : alpha_b);
                  __m256 a = _mm256_set1_ps(a_val);
                  __m256 new_t = _mm256_fmadd_ps(a, p_c, t);
                  new_t = _mm256_blendv_ps(t, new_t, mask_ps);
                  scatter_channel(tgt, off_c, new_t);

                  __m256 old_r = gather_channel(r, off_c);
                  __m256 new_r = _mm256_fnmadd_ps(a, ap, old_r);
                  new_r = _mm256_and_ps(new_r, mask_ps);
                  scatter_channel(r, off_c, new_r);

                  __m256 r2 = _mm256_mul_ps(new_r, new_r);
                  float tmp[8];
                  _mm256_storeu_ps(tmp, r2);
                  double sum = 0;
                  for (int k = 0; k < 8; ++k) sum += tmp[k];
                  
                  if(c == 0) local_new_r += sum;
                  if(c == 1) local_new_g += sum;
                  if(c == 2) local_new_b += sum;
              }
          }
          for (int x = ((M-2) & ~7) + 1; x <= M-2; ++x) {
              int id = y * M + x;
              if (mask[id]) {
                  int off3 = id * 3;
                  
                  tgt[off3+0] += alpha_r * p[off3+0];
                  r[off3+0] -= alpha_r * Ap[off3+0];
                  local_new_r += r[off3+0] * r[off3+0];
                  
                  tgt[off3+1] += alpha_g * p[off3+1];
                  r[off3+1] -= alpha_g * Ap[off3+1];
                  local_new_g += r[off3+1] * r[off3+1];
                  
                  tgt[off3+2] += alpha_b * p[off3+2];
                  r[off3+2] -= alpha_b * Ap[off3+2];
                  local_new_b += r[off3+2] * r[off3+2];
              }
          }
          r_sq_new_r += local_new_r;
          r_sq_new_g += local_new_g;
          r_sq_new_b += local_new_b;
      }
#else
      #pragma omp parallel for reduction(+:r_sq_new_r, r_sq_new_g, r_sq_new_b) schedule(static)
      for (int y = 1; y < N - 1; ++y) {
        for (int x = 1; x < M - 1; ++x) {
          int id = y * M + x;
          if (mask[id]) {
            int off3 = id * 3;
            tgt[off3+0] += alpha_r * p[off3+0];
            tgt[off3+1] += alpha_g * p[off3+1];
            tgt[off3+2] += alpha_b * p[off3+2];

            r[off3+0] -= alpha_r * Ap[off3+0];
            r[off3+1] -= alpha_g * Ap[off3+1];
            r[off3+2] -= alpha_b * Ap[off3+2];

            r_sq_new_r += r[off3+0]*r[off3+0];
            r_sq_new_g += r[off3+1]*r[off3+1];
            r_sq_new_b += r[off3+2]*r[off3+2];
          }
        }
      }
#endif

      float beta_r = (r_sq_r > 1e-12) ? (r_sq_new_r / r_sq_r) : 0.0f;
      float beta_g = (r_sq_g > 1e-12) ? (r_sq_new_g / r_sq_g) : 0.0f;
      float beta_b = (r_sq_b > 1e-12) ? (r_sq_new_b / r_sq_b) : 0.0f;

      r_sq_r = r_sq_new_r;
      r_sq_g = r_sq_new_g;
      r_sq_b = r_sq_new_b;

#ifdef __AVX2__
      // 3. Update p
      #pragma omp parallel for schedule(static)
      for (int y = 1; y < N - 1; ++y) {
          for (int x = 1; x <= M - 2; x += 8) {
              int xend = (x + 7 <= M - 2) ? 8 : (M - 1 - x);
              if (xend < 8) break;

              for (int c = 0; c < 3; ++c) {
                  __m256i off_c = make_offset_vec(y, x, c, M);
                  __m256i mask32 = load_mask(mask, y, x, M);
                  __m256 mask_ps = _mm256_castsi256_ps(mask32);

                  __m256 old_p = gather_channel(p, off_c);
                  __m256 res   = gather_channel(r, off_c);
                  
                  float b_val = (c == 0) ? beta_r : ((c == 1) ? beta_g : beta_b);
                  __m256 b = _mm256_set1_ps(b_val);

                  __m256 new_p = _mm256_fmadd_ps(b, old_p, res);
                  new_p = _mm256_and_ps(new_p, mask_ps);
                  scatter_channel(p, off_c, new_p);
              }
          }
          for (int x = ((M-2) & ~7) + 1; x <= M-2; ++x) {
              int id = y * M + x;
              if (mask[id]) {
                  int off3 = id * 3;
                  p[off3+0] = r[off3+0] + beta_r * p[off3+0];
                  p[off3+1] = r[off3+1] + beta_g * p[off3+1];
                  p[off3+2] = r[off3+2] + beta_b * p[off3+2];
              }
          }
      }
#else
      #pragma omp parallel for schedule(static)
      for (int y = 1; y < N - 1; ++y) {
        for (int x = 1; x < M - 1; ++x) {
          int id = y * M + x;
          if (mask[id]) {
            int off3 = id * 3;
            p[off3+0] = r[off3+0] + beta_r * p[off3+0];
            p[off3+1] = r[off3+1] + beta_g * p[off3+1];
            p[off3+2] = r[off3+2] + beta_b * p[off3+2];
          }
        }
      }
#endif

      // Check real convergence every 50 iterations to avoid slowdown
      if (it % 50 == 0) {
          calc_error();
          current_err = std::max({err[0], err[1], err[2]});
      }
      
      it++;
  } 

  LIKWID_MARKER_STOP("v5_compute");

  calc_error();

  // Clamp output
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N * M * 3; ++i) {
    imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : tgt[i];
  }

  return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}
