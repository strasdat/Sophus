#include <iostream>

#include <sophus/so2.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::SO2<double>>;
template class Map<Sophus::SO2<double> const>;
}

namespace Sophus {

template class SO2<double>;

template <class Scalar>
void tests() {
  using std::vector;
  using SO2Type = SO2<Scalar>;
  using Point = typename SO2<Scalar>::Point;
  using Tangent = typename SO2<Scalar>::Tangent;
  Scalar const PI = Constants<Scalar>::pi();

  vector<SO2Type, Eigen::aligned_allocator<SO2Type>> so2_vec;
  so2_vec.push_back(SO2Type::exp(0.0));
  so2_vec.push_back(SO2Type::exp(0.2));
  so2_vec.push_back(SO2Type::exp(10.));
  so2_vec.push_back(SO2Type::exp(0.00001));
  so2_vec.push_back(SO2Type::exp(PI));
  so2_vec.push_back(SO2Type::exp(0.2) * SO2Type::exp(PI) * SO2Type::exp(-0.2));
  so2_vec.push_back(SO2Type::exp(-0.3) * SO2Type::exp(PI) * SO2Type::exp(0.3));

  vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec;
  tangent_vec.push_back(Tangent(0));
  tangent_vec.push_back(Tangent(1));
  tangent_vec.push_back(Tangent(PI / 2.));
  tangent_vec.push_back(Tangent(-1));
  tangent_vec.push_back(Tangent(20));
  tangent_vec.push_back(Tangent(PI / 2. + 0.0001));

  vector<Point, Eigen::aligned_allocator<Point>> point_vec;
  point_vec.push_back(Point(1, 2));

  LieGroupTests<SO2Type> tests;
  tests.setGroupElements(so2_vec);
  tests.setTangentVectors(tangent_vec);
  tests.setPoints(point_vec);

  bool passed = tests.doAllTestsPass();

  // Test that the complex number magnitude stays close to one.
  SO2Type current_z;
  for (std::size_t i = 0; i < 1000; ++i) {
    for (SO2Type const& z : so2_vec) {
      current_z *= z;
    }
  }
  SOPHUS_TEST_APPROX(passed, current_z.unit_complex().norm(), Scalar(1),
                     Constants<Scalar>::epsilon(), "Magnitude drift");
  processTestResult(passed);
}

int test_so2() {
  using std::cerr;
  using std::endl;

  cerr << "Test SO2" << endl << endl;
  cerr << "Double tests: " << endl;
  tests<double>();
  cerr << "Float tests: " << endl;
  tests<float>();
  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_so2(); }
