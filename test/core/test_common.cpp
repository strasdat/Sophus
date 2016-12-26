#include <iostream>

#include <sophus/test_macros.hpp>

namespace Sophus {

namespace {

bool testFormatString() {
  bool passed = true;
  SOPHUS_TEST_EQUAL(passed, details::FormatString(), std::string());
  std::string test_str = "Hello World!";
  SOPHUS_TEST_EQUAL(passed, details::FormatString(test_str.c_str()), test_str);
  SOPHUS_TEST_EQUAL(passed, details::FormatString("Number: %", 5),
                    std::string("Number: 5"));
  SOPHUS_TEST_EQUAL(passed,
                    details::FormatString("Real: % msg %", 1.5, test_str),
                    std::string("Real: 1.5 msg Hello World!"));
  SOPHUS_TEST_EQUAL(passed,
                    details::FormatString(
                        "vec: %", Eigen::Vector3f(0.f, 1.f, 1.5f).transpose()),
                    std::string("vec:   0   1 1.5"));
  SOPHUS_TEST_EQUAL(
      passed, details::FormatString("Number: %", 1, 2),
      std::string("Number: 1\nFormat-Warning: There are 1 args unused."));
  return passed;
}

void runAll() {
  std::cerr << "Common tests:" << std::endl;
  bool passed = testFormatString();
  processTestResult(passed);
}

}  // namespace
}  // namespace Sophus

int main() { Sophus::runAll(); }
