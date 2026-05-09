#include "solver.h"
#include "likwid_wrapper.h"
#include <stdlib.h>

static bool likwid_init_done = false;

PYBIND11_MODULE(core_openmp, m) {
  m.def("likwid_init", []() {
    if (!likwid_init_done) {
      LIKWID_MARKER_INIT;
      likwid_init_done = true;
    }
  });
  m.def("likwid_close", []() {
    if (likwid_init_done) {
      const char* filepath = getenv("LIKWID_FILEPATH");
      if (filepath) {
        LIKWID_MARKER_WRITE_FILE(filepath);
      }
      LIKWID_MARKER_CLOSE;
      likwid_init_done = false;
    }
  });

  py::class_<OpenMPEquSolver>(m, "EquSolver")
      .def(py::init<int>())
      .def("partition", &OpenMPEquSolver::partition)
      .def("reset", &OpenMPEquSolver::reset)
      .def("sync", &OpenMPEquSolver::sync)
      .def("step", &OpenMPEquSolver::step);
  py::class_<OpenMPGridSolver>(m, "GridSolver")
      .def(py::init<int, int, int>())
      .def("reset", &OpenMPGridSolver::reset)
      .def("sync", &OpenMPGridSolver::sync)
      .def("step", &OpenMPGridSolver::step);
  py::class_<OpenMPMultigridSolver>(m, "MultigridSolver")
      .def(py::init<int, int, int>())
      .def("reset", &OpenMPMultigridSolver::reset)
      .def("sync", &OpenMPMultigridSolver::sync)
      .def("step", &OpenMPMultigridSolver::step);
  py::class_<OpenMPSolverV3>(m, "SolverV3")
      .def(py::init<int, int, int>())
      .def("reset", &OpenMPSolverV3::reset)
      .def("sync", &OpenMPSolverV3::sync)
      .def("step", &OpenMPSolverV3::step);
  py::class_<OpenMPSolverV4>(m, "SolverV4")
      .def(py::init<int, int, int>())
      .def("reset", &OpenMPSolverV4::reset)
      .def("sync", &OpenMPSolverV4::sync)
      .def("step", &OpenMPSolverV4::step);
  py::class_<OpenMPSolverV5>(m, "SolverV5")
      .def(py::init<int, int, int>())
      .def("reset", &OpenMPSolverV5::reset)
      .def("sync", &OpenMPSolverV5::sync)
      .def("step", &OpenMPSolverV5::step);
}
