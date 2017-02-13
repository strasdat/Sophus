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
class Tests {
 public:
  using SO2Type = SO2<Scalar>;
  using Point = typename SO2<Scalar>::Point;
  using Tangent = typename SO2<Scalar>::Tangent;
  Scalar const PI = Constants<Scalar>::pi();

  Tests() {
    so2_vec_.push_back(SO2Type::exp(0.0));
    so2_vec_.push_back(SO2Type::exp(0.2));
    so2_vec_.push_back(SO2Type::exp(10.));
    so2_vec_.push_back(SO2Type::exp(0.00001));
    so2_vec_.push_back(SO2Type::exp(PI));
    so2_vec_.push_back(SO2Type::exp(0.2) * SO2Type::exp(PI) *
                       SO2Type::exp(-0.2));
    so2_vec_.push_back(SO2Type::exp(-0.3) * SO2Type::exp(PI) *
                       SO2Type::exp(0.3));

    tangent_vec_.push_back(Tangent(0));
    tangent_vec_.push_back(Tangent(1));
    tangent_vec_.push_back(Tangent(PI / 2.));
    tangent_vec_.push_back(Tangent(-1));
    tangent_vec_.push_back(Tangent(20));
    tangent_vec_.push_back(Tangent(PI / 2. + 0.0001));

    point_vec_.push_back(Point(1, 2));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testUnity();
    passed &= testRawDataAcces();
    passed &= testConstructors();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<SO2Type> tests;
    tests.setGroupElements(so2_vec_);
    tests.setTangentVectors(tangent_vec_);
    tests.setPoints(point_vec_);
    return tests.doAllTestsPass();
  }

  bool testUnity() {
    bool passed = true;
    // Test that the complex number magnitude stays close to one.
    SO2Type current_q;
    for (std::size_t i = 0; i < 1000; ++i) {
      for (SO2Type const& q : so2_vec_) {
        current_q *= q;
      }
    }
    SOPHUS_TEST_APPROX(passed, current_q.unit_complex().norm(), Scalar(1),
                       Constants<Scalar>::epsilon(), "Magnitude drift");
    return passed;
  }

  bool testRawDataAcces() {
    bool passed = true;
    Vector2<Scalar> raw = {0, 1};
    Eigen::Map<SO2Type const> const_so2_map(raw.data());
    SOPHUS_TEST_APPROX(passed, const_so2_map.unit_complex().eval(), raw,
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, const_so2_map.unit_complex().data(), raw.data());
    Eigen::Map<SO2Type const> const_shallow_copy = const_so2_map;
    SOPHUS_TEST_EQUAL(passed, const_shallow_copy.unit_complex().eval(),
                      const_so2_map.unit_complex().eval());

    Vector2<Scalar> raw2 = {1, 0};
    Eigen::Map<SO2Type> so2_map(raw.data());
    so2_map.setComplex(raw2);
    SOPHUS_TEST_APPROX(passed, so2_map.unit_complex().eval(), raw2,
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, so2_map.unit_complex().data(), raw.data());
    SOPHUS_TEST_NEQ(passed, so2_map.unit_complex().data(), raw2.data());
    Eigen::Map<SO2Type> shallow_copy = so2_map;
    SOPHUS_TEST_EQUAL(passed, shallow_copy.unit_complex().eval(),
                      so2_map.unit_complex().eval());

    SO2Type const const_so2(raw2);
    for (int i = 0; i < 2; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_so2.data()[i], raw2.data()[i]);
    }

    SO2Type so2(raw2);
    for (int i = 0; i < 2; ++i) {
      so2.data()[i] = raw[i];
    }

    for (int i = 0; i < 2; ++i) {
      SOPHUS_TEST_EQUAL(passed, so2.data()[i], raw.data()[i]);
    }
    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Matrix2<Scalar> R = so2_vec_.front().matrix();
    SO2Type so2(R);
    SOPHUS_TEST_APPROX(passed, R, so2.matrix(), Constants<Scalar>::epsilon());
    return passed;
  }

  std::vector<SO2Type, Eigen::aligned_allocator<SO2Type>> so2_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int test_so2() {
  using std::cerr;
  using std::endl;

  cerr << "Test SO2" << endl << endl;
  cerr << "Double tests: " << endl;
  Tests<double>().runAll();
  cerr << "Float tests: " << endl;
  Tests<float>().runAll();
  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_so2(); }
