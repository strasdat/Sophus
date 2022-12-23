// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/so2.h"

#include "sophus/lie/details/test_impl.h"

#include <iostream>

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {  // NOLINT
template class Map<sophus::So2<double>>;
template class Map<sophus::So2<double> const>;
}  // namespace Eigen

namespace sophus {

template class So2<double>;
#if SOPHUS_CERES
template class So2<ceres::Jet<double, 3>>;
#endif

template <class TScalar>
class Tests {
 public:
  using Scalar = TScalar;
  using So2Type = So2<Scalar>;
  using Point = typename So2<Scalar>::Point;
  using Tangent = typename So2<Scalar>::Tangent;
  Scalar const k_pi = kPi<Scalar>;  // NOLINT

  Tests() {
    so2_vec_.push_back(So2Type::exp(Scalar(0.0)));
    so2_vec_.push_back(So2Type::exp(Scalar(0.2)));
    so2_vec_.push_back(So2Type::exp(Scalar(10.)));
    so2_vec_.push_back(So2Type::exp(Scalar(0.00001)));
    so2_vec_.push_back(So2Type::exp(k_pi));
    so2_vec_.push_back(
        So2Type::exp(Scalar(0.2)) * So2Type::exp(k_pi) *
        So2Type::exp(Scalar(-0.2)));
    so2_vec_.push_back(
        So2Type::exp(Scalar(-0.3)) * So2Type::exp(k_pi) *
        So2Type::exp(Scalar(0.3)));

    tangent_vec_.push_back(Tangent(Scalar(0)));
    tangent_vec_.push_back(Tangent(Scalar(1)));
    tangent_vec_.push_back(Tangent(Scalar(k_pi / 2.)));
    tangent_vec_.push_back(Tangent(Scalar(-1)));
    tangent_vec_.push_back(Tangent(Scalar(20)));
    tangent_vec_.push_back(Tangent(Scalar(k_pi / 2. + 0.0001)));

    point_vec_.push_back(Point(Scalar(1), Scalar(2)));
    point_vec_.push_back(Point(Scalar(1), Scalar(-3)));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testUnity();
    passed &= testRawDataAcces();
    passed &= testConstructors();
    passed &= testFit();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<So2Type> tests(so2_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testUnity() {
    bool passed = true;
    // Test that the complex number magnitude stays close to one.
    So2Type current_q;
    for (std::size_t i = 0; i < 1000; ++i) {
      for (So2Type const& q : so2_vec_) {
        current_q *= q;
      }
    }
    SOPHUS_TEST_APPROX(
        passed,
        current_q.unitComplex().norm(),
        Scalar(1),
        kEpsilon<Scalar>,
        "Magnitude drift");
    return passed;
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Vector2<Scalar> raw = {0, 1};
    Eigen::Map<So2Type const> map_of_const_so2(raw.data());
    SOPHUS_TEST_APPROX(
        passed,
        map_of_const_so2.unitComplex().eval(),
        raw,
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_const_so2.unitComplex().data(), raw.data(), "");
    Eigen::Map<So2Type const> const_shallow_copy = map_of_const_so2;
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.unitComplex().eval(),
        map_of_const_so2.unitComplex().eval(),
        "");

    Eigen::Vector2<Scalar> raw2 = {1, 0};
    Eigen::Map<So2Type> map_of_so2(raw.data());
    map_of_so2.setComplex(raw2);
    SOPHUS_TEST_APPROX(
        passed, map_of_so2.unitComplex().eval(), raw2, kEpsilon<Scalar>, "");
    SOPHUS_TEST_EQUAL(passed, map_of_so2.unitComplex().data(), raw.data(), "");
    SOPHUS_TEST_NEQ(passed, map_of_so2.unitComplex().data(), raw2.data(), "");
    Eigen::Map<So2Type> shallow_copy = map_of_so2;
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.unitComplex().eval(),
        map_of_so2.unitComplex().eval(),
        "");

    So2Type const const_so2 = So2Type::fromParams(raw2);
    for (int i = 0; i < 2; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_so2.data()[i], raw2.data()[i], "");
    }

    So2Type so2 = So2Type::fromParams(raw2);
    for (int i = 0; i < 2; ++i) {
      so2.data()[i] = raw[i];
    }

    for (int i = 0; i < 2; ++i) {
      SOPHUS_TEST_EQUAL(passed, so2.data()[i], raw.data()[i], "");
    }

    Eigen::Vector2<Scalar> data1 = {1, 0};

    Eigen::Vector2<Scalar> data2 = {0, 1};
    Eigen::Map<So2Type> map1(data1.data());

    Eigen::Map<So2Type> map2(data2.data());

    // map -> map assignment
    map2 = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), map2.matrix(), "");

    // map -> type assignment
    So2Type copy;
    copy = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    // type -> map assignment
    copy = So2Type::fromAngle(Scalar(0.5));
    map1 = copy;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Eigen::Matrix2<Scalar> r = so2_vec_.front().matrix();
    So2Type so2 = So2Type::fromMatrix(r);
    SOPHUS_TEST_APPROX(passed, r, so2.matrix(), kEpsilon<Scalar>, "");

    return passed;
  }

  template <class TS = Scalar>
  std::enable_if_t<std::is_floating_point<TS>::value, bool> testFit() {
    bool passed = true;

    for (int i = 0; i < 100; ++i) {
      Eigen::Matrix2<Scalar> r = Eigen::Matrix2<Scalar>::Random();
      So2Type so2 = So2Type::fitToSo2(r);
      So2Type so2_2 = So2Type::fitToSo2(so2.matrix());

      SOPHUS_TEST_APPROX(
          passed, so2.matrix(), so2_2.matrix(), kEpsilon<Scalar>, "");
    }
    return passed;
  }

  template <class TS = Scalar>
  std::enable_if_t<!std::is_floating_point<TS>::value, bool> testFit() {
    return true;
  }

  std::vector<So2Type, Eigen::aligned_allocator<So2Type>> so2_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int testSo2() {
  using std::cerr;
  using std::endl;

  cerr << "Test So2" << endl << endl;
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
}  // namespace sophus

int main() { return sophus::testSo2(); }
