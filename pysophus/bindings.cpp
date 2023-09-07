#include "SophusPyBind.h"

PYBIND11_MODULE(pysophus, m) {
  sophus::exportSophus(m);
}
