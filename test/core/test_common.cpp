#include <iostream>

#include <sophus/test_macros.hpp>

namespace Sophus {

namespace {

bool testSmokeDetails() {
  bool passed = true;
  std::cout << details::pretty(4.2) << std::endl;
  std::cout << details::pretty(Vector2f(1, 2)) << std::endl;
  bool dummy = true;
  details::testFailed(dummy, "dummyFunc", "dummyFile", 99,
                      "This is just a pratice alarm!");
  SOPHUS_TEST_EQUAL(passed, dummy, false, "");

  double val = transpose(42.0);
  SOPHUS_TEST_EQUAL(passed, val, 42.0, "");
  Matrix<float, 1, 2> row = transpose(Vector2f(1, 7));
  Matrix<float, 1, 2> expected_row(1, 7);
  SOPHUS_TEST_EQUAL(passed, row, expected_row, "");

  optional<int> opt(nullopt);
  SOPHUS_TEST(passed, !opt, "");

  return passed;
}

void runAll() {
  std::cerr << "Common tests:" << std::endl;
  bool passed = testSmokeDetails();
  processTestResult(passed);
}

}  // namespace
}  // namespace Sophus

int main() { Sophus::runAll(); }
