#include <iostream>

#include <sophus/cartesian.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::Cartesian<double, 3>>;
template class Map<Sophus::Cartesian<double, 3> const>;
}  // namespace Eigen

namespace Sophus {

template class Cartesian<double, 3, Eigen::AutoAlign>;
template class Cartesian<float, 3, Eigen::DontAlign>;
#if SOPHUS_CERES
template class Cartesian<ceres::Jet<double, 3>, 3>;
#endif

template <class Scalar_>
class Tests {
 public:
  using Scalar = Scalar_;
  using Point = typename Cartesian<Scalar, 3>::Point;
  using Tangent = typename Cartesian<Scalar, 3>::Tangent;

  Tests() {
    Tangent tmp;
    tmp << Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(1), Scalar(0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(0), Scalar(-0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(-1), Scalar(1), Scalar(-0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(20), Scalar(-1), Scalar(2);
    tangent_vec_.push_back(tmp);

    for (const auto& t : tangent_vec_) {
      cartesian3_vec_.push_back(Cartesian<Scalar, 3>(t));
    }

    point_vec_.push_back(Point(Scalar(1), Scalar(2), Scalar(4)));
    point_vec_.push_back(Point(Scalar(1), Scalar(-3), Scalar(0.5)));
    point_vec_.push_back(Point(Scalar(-5), Scalar(-6), Scalar(7)));
  }

  void runAll() {
    bool passed = testLieProperties();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<Cartesian<Scalar, 3>> tests(cartesian3_vec_, tangent_vec_,
                                              point_vec_);
    return tests.doAllTestsPass();
  }

  std::vector<Cartesian<Scalar, 3>,
              Eigen::aligned_allocator<Cartesian<Scalar, 3>>>
      cartesian3_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int test_cartesian() {
  using std::cerr;
  using std::endl;

  cerr << "Test Cartesian3" << endl << endl;
  cerr << "Double tests: " << endl;
  Tests<double>().runAll();
  cerr << "Float tests: " << endl;
  Tests<float>().runAll();

#if SOPHUS_CERES
  cerr << "ceres::Jet<double, 3> tests: " << endl;
  Tests<ceres::Jet<double, 3>>().runAll();
#endif

  return 0;
}

}  // namespace Sophus

int main() { return Sophus::test_cartesian(); }
