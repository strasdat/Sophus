#include <iostream>

#include <sophus/se2.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::SE2<double>>;
template class Map<Sophus::SE2<double> const>;
}

namespace Sophus {

template class SE2<double>;

template <class Scalar>
class Tests {
 public:
  using SE2Type = SE2<Scalar>;
  using SO2Type = SO2<Scalar>;
  using Point = typename SE2<Scalar>::Point;
  using Tangent = typename SE2<Scalar>::Tangent;
  Scalar const kPi = Constants<Scalar>::pi();

  Tests() {
    se2_vec_.push_back(SE2Type(SO2Type(0.0), Point(0, 0)));
    se2_vec_.push_back(SE2Type(SO2Type(0.2), Point(10, 0)));
    se2_vec_.push_back(SE2Type(SO2Type(0.), Point(0, 100)));
    se2_vec_.push_back(SE2Type(SO2Type(-1.), Point(20, -1)));
    se2_vec_.push_back(
        SE2Type(SO2Type(0.00001), Point(-0.00000001, 0.0000000001)));
    se2_vec_.push_back(SE2Type(SO2Type(0.2), Point(0, 0)) *
                       SE2Type(SO2Type(kPi), Point(0, 0)) *
                       SE2Type(SO2Type(-0.2), Point(0, 0)));
    se2_vec_.push_back(SE2Type(SO2Type(0.3), Point(2, 0)) *
                       SE2Type(SO2Type(kPi), Point(0, 0)) *
                       SE2Type(SO2Type(-0.3), Point(0, 6)));

    Tangent tmp;
    tmp << 0, 0, 0;
    tangent_vec_.push_back(tmp);
    tmp << 1, 0, 0;
    tangent_vec_.push_back(tmp);
    tmp << 0, 1, 1;
    tangent_vec_.push_back(tmp);
    tmp << -1, 1, 0;
    tangent_vec_.push_back(tmp);
    tmp << 20, -1, -1;
    tangent_vec_.push_back(tmp);
    tmp << 30, 5, 20;
    tangent_vec_.push_back(tmp);

    point_vec_.push_back(Point(1, 2));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testRawDataAcces();
    passed &= testConstructors();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<SE2Type> tests(se2_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 4, 1> raw;
    raw << 0, 1, 0, 3;
    Eigen::Map<SE2Type const> const_se2_map(raw.data());
    SOPHUS_TEST_APPROX(passed, const_se2_map.unit_complex().eval(),
                       raw.template head<2>().eval(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, const_se2_map.translation().eval(),
                       raw.template tail<2>().eval(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, const_se2_map.unit_complex().data(), raw.data());
    SOPHUS_TEST_EQUAL(passed, const_se2_map.translation().data(),
                      raw.data() + 2);
    Eigen::Map<SE2Type const> const_shallow_copy = const_se2_map;
    SOPHUS_TEST_EQUAL(passed, const_shallow_copy.unit_complex().eval(),
                      const_se2_map.unit_complex().eval());
    SOPHUS_TEST_EQUAL(passed, const_shallow_copy.translation().eval(),
                      const_se2_map.translation().eval());

    Eigen::Matrix<Scalar, 4, 1> raw2;
    raw2 << 1, 0, 3, 1;
    Eigen::Map<SE2Type> map_of_se3(raw.data());
    map_of_se3.setComplex(raw2.template head<2>());
    map_of_se3.translation() = raw2.template tail<2>();
    SOPHUS_TEST_APPROX(passed, map_of_se3.unit_complex().eval(),
                       raw2.template head<2>().eval(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, map_of_se3.translation().eval(),
                       raw2.template tail<2>().eval(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, map_of_se3.unit_complex().data(), raw.data());
    SOPHUS_TEST_EQUAL(passed, map_of_se3.translation().data(), raw.data() + 2);
    SOPHUS_TEST_NEQ(passed, map_of_se3.unit_complex().data(), raw2.data());
    Eigen::Map<SE2Type> shallow_copy = map_of_se3;
    SOPHUS_TEST_EQUAL(passed, shallow_copy.unit_complex().eval(),
                      map_of_se3.unit_complex().eval());
    SOPHUS_TEST_EQUAL(passed, shallow_copy.translation().eval(),
                      map_of_se3.translation().eval());
    Eigen::Map<SE2Type> const const_map_of_se2 = map_of_se3;
    SOPHUS_TEST_EQUAL(passed, const_map_of_se2.unit_complex().eval(),
                      map_of_se3.unit_complex().eval());
    SOPHUS_TEST_EQUAL(passed, const_map_of_se2.translation().eval(),
                      map_of_se3.translation().eval());

    SE2Type const const_se2(raw2.template head<2>().eval(),
                            raw2.template tail<2>().eval());
    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_se2.data()[i], raw2.data()[i]);
    }

    SE2Type se2(raw2.template head<2>().eval(), raw2.template tail<2>().eval());
    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, se2.data()[i], raw2.data()[i]);
    }

    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, se2.data()[i], raw.data()[i]);
    }
    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Matrix3<Scalar> I = Matrix3<Scalar>::Identity();
    SOPHUS_TEST_EQUAL(passed, SE2Type().matrix(), I);

    SE2Type se2 = se2_vec_.front();
    Point translation = se2.translation();
    SO2Type so2 = se2.so2();

    SOPHUS_TEST_APPROX(passed, SE2Type(so2.log(), translation).matrix(),
                       se2.matrix(), Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, SE2Type(so2, translation).matrix(), se2.matrix(),
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, SE2Type(so2.matrix(), translation).matrix(),
                       se2.matrix(), Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed,
                       SE2Type(so2.unit_complex(), translation).matrix(),
                       se2.matrix(), Constants<Scalar>::epsilon());
    SOPHUS_TEST_APPROX(passed, SE2Type(se2.matrix()).matrix(), se2.matrix(),
                       Constants<Scalar>::epsilon());
    return passed;
  }

  std::vector<SE2Type, Eigen::aligned_allocator<SE2Type>> se2_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int test_se2() {
  using std::cerr;
  using std::endl;

  cerr << "Test SE2" << endl << endl;
  cerr << "Double tests: " << endl;
  Tests<double>().runAll();
  cerr << "Float tests: " << endl;
  Tests<float>().runAll();
  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_se2(); }
