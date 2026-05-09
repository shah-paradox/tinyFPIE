#ifndef FPIE_CORE_OPENMP_SOLVER_H_
#define FPIE_CORE_OPENMP_SOLVER_H_

#include <tuple>

#include "base_solver.h"

class OpenMPEquSolver : public EquSolver {
  int* maskbuf;
  unsigned char* imgbuf;
  float* tmp;
  int n_mid;

 public:
  explicit OpenMPEquSolver(int n_cpu);
  ~OpenMPEquSolver();

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();

  inline void update_equation(int i);

  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

class OpenMPGridSolver : public GridSolver {
  unsigned char* imgbuf;
  float* tmp;

 public:
  OpenMPGridSolver(int grid_x, int grid_y, int n_cpu);
  ~OpenMPGridSolver();

  void post_reset();

  inline void update_equation(int id);

  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

class OpenMPMultigridSolver : public GridSolver {
  unsigned char* imgbuf;
  float* tmp;
  int* mask_c;
  float* tgt_c;
  float* grad_c;
  int N_c;
  int M_c;
  int m3_c;

 public:
  OpenMPMultigridSolver(int grid_x, int grid_y, int n_cpu);
  ~OpenMPMultigridSolver();

  void post_reset();
  inline void smooth(int n_rows, int n_cols, float* t, const float* g, const int* msk, int m3_val);
  void restrict_residual();
  void prolongate_correction();
  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

class OpenMPSolverV3 : public GridSolver {
  unsigned char* imgbuf;

 public:
  OpenMPSolverV3(int grid_x, int grid_y, int n_cpu);
  ~OpenMPSolverV3();

  void post_reset();
  void calc_error();
  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

class OpenMPSolverV4 : public GridSolver {
  unsigned char* imgbuf;
  float *r;
  float *p;
  float *Ap;

 public:
  OpenMPSolverV4(int grid_x, int grid_y, int n_cpu);
  ~OpenMPSolverV4();

  void post_reset();
  void calc_error();
  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

class OpenMPSolverV5 : public GridSolver {
  unsigned char* imgbuf;
  float *r;
  float *p;
  float *Ap;

 public:
  OpenMPSolverV5(int grid_x, int grid_y, int n_cpu);
  ~OpenMPSolverV5();

  void post_reset();
  void calc_error();
  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int max_iterations);
};

#endif  // FPIE_CORE_OPENMP_SOLVER_H_
