#include "SophusPyBind.h"

PYBIND11_MODULE(pysophus, m) { Sophus::exportSophus(m); }
