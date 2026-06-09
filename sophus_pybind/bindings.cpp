#include <sophus/interpolate.hpp>
#include "SophusPyBind.h"

namespace py = pybind11;

PYBIND11_MODULE(sophus_pybind, m) {
  Sophus::exportSophus(m);

  m.def(
      "interpolate",
      [](Sophus::SE3d const& a, Sophus::SE3d const& b, double t) {
        return Sophus::interpolate(a, b, t);
      },
      py::arg("a"), py::arg("b"), py::arg("t"),
      "Interpolates between two SE3 poses");

  m.def(
      "interpolate",
      [](Sophus::SO3d const& a, Sophus::SO3d const& b, double t) {
        return Sophus::interpolate(a, b, t);
      },
      py::arg("a"), py::arg("b"), py::arg("t"),
      "Interpolates between two SO3 rotations");

  m.def(
      "interpolate",
      [](Sophus::SE2d const& a, Sophus::SE2d const& b, double t) {
        return Sophus::interpolate(a, b, t);
      },
      py::arg("a"), py::arg("b"), py::arg("t"),
      "Interpolates between two SE2 poses");

  m.def(
      "interpolate",
      [](Sophus::SO2d const& a, Sophus::SO2d const& b, double t) {
        return Sophus::interpolate(a, b, t);
      },
      py::arg("a"), py::arg("b"), py::arg("t"),
      "Interpolates between two SO2 rotations");
}
