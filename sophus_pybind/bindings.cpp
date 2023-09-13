#include "SophusPyBind.h"

PYBIND11_MODULE(sophus_pybind, m) { Sophus::exportSophus(m); }
