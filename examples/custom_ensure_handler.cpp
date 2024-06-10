#include <sophus/common.hpp>

#include <cstdio>
#include <cstdlib>

namespace Sophus {
void ensureFailed(char const* function, char const* file, int line,
                  char const* description) {
  std::printf("Sophus ensure failed in function '%s', file '%s', line %d.\n",
              file, function, line);
  throw std::runtime_error("Sophus ensure failed");
}
}  // namespace Sophus
