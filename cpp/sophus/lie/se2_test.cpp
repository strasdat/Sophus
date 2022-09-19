// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/se2.h"

#include "sophus/lie/details/test_impl.h"

#include <unsupported/Eigen/MatrixFunctions>

#include <iostream>

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {  // NOLINT
template class Map<sophus::Se2<double>>;
template class Map<sophus::Se2<double> const>;
}  // namespace Eigen

namespace sophus {

template class Se2<double, Eigen::AutoAlign>;
template class Se2<double, Eigen::DontAlign>;
#if SOPHUS_CERES
template class Se2<ceres::Jet<double, 3>>;
#endif

template <class ScalarT>
class Tests {
 public:
  using Scalar = ScalarT;
  using SE2Type = Se2<Scalar>;
  using SO2Type = So2<Scalar>;
  using Point = typename Se2<Scalar>::Point;
  using Tangent = typename Se2<Scalar>::Tangent;
  Scalar const k_pi = kPi<Scalar>;  // NOLINT

  Tests() {
    se2_vec_.push_back(
        SE2Type(SO2Type(Scalar(0.0)), Point(Scalar(0), Scalar(0))));
    se2_vec_.push_back(
        SE2Type(SO2Type(Scalar(0.2)), Point(Scalar(10), Scalar(0))));
    se2_vec_.push_back(
        SE2Type(SO2Type(Scalar(0.)), Point(Scalar(0), Scalar(100))));
    se2_vec_.push_back(
        SE2Type(SO2Type(Scalar(-1.)), Point(Scalar(20), -Scalar(1))));
    se2_vec_.push_back(SE2Type(
        SO2Type(Scalar(0.00001)),
        Point(Scalar(-0.00000001), Scalar(0.0000000001))));
    se2_vec_.push_back(
        SE2Type(SO2Type(Scalar(0.2)), Point(Scalar(0), Scalar(0))) *
        SE2Type(SO2Type(k_pi), Point(Scalar(0), Scalar(0))) *
        SE2Type(SO2Type(Scalar(-0.2)), Point(Scalar(0), Scalar(0))));
    se2_vec_.push_back(
        SE2Type(SO2Type(Scalar(0.3)), Point(Scalar(2), Scalar(0))) *
        SE2Type(SO2Type(k_pi), Point(Scalar(0), Scalar(0))) *
        SE2Type(SO2Type(Scalar(-0.3)), Point(Scalar(0), Scalar(6))));

    Tangent tmp;
    tmp << Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(1), Scalar(1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(-1), Scalar(1), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(20), Scalar(-1), Scalar(-1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(30), Scalar(5), Scalar(20);
    tangent_vec_.push_back(tmp);

    point_vec_.push_back(Point(1, 2));
    point_vec_.push_back(Point(1, -3));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testRawDataAcces();
    passed &= testMutatingAccessors();
    passed &= testConstructors();
    passed &= testFit();
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
    raw << Scalar(0), Scalar(1), Scalar(0), Scalar(3);
    Eigen::Map<SE2Type const> const_se2_map(raw.data());
    SOPHUS_TEST_APPROX(
        passed,
        const_se2_map.unitComplex().eval(),
        raw.template head<2>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        const_se2_map.translation().eval(),
        raw.template tail<2>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(
        passed, const_se2_map.unitComplex().data(), raw.data(), "");
    SOPHUS_TEST_EQUAL(
        passed, const_se2_map.translation().data(), raw.data() + 2, "");
    Eigen::Map<SE2Type const> const_shallow_copy = const_se2_map;
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.unitComplex().eval(),
        const_se2_map.unitComplex().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.translation().eval(),
        const_se2_map.translation().eval(),
        "");

    Eigen::Matrix<Scalar, 4, 1> raw2;
    raw2 << Scalar(1), Scalar(0), Scalar(3), Scalar(1);
    Eigen::Map<SE2Type> map_of_se3(raw.data());
    map_of_se3.setComplex(raw2.template head<2>());
    map_of_se3.translation() = raw2.template tail<2>();
    SOPHUS_TEST_APPROX(
        passed,
        map_of_se3.unitComplex().eval(),
        raw2.template head<2>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        map_of_se3.translation().eval(),
        raw2.template tail<2>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(passed, map_of_se3.unitComplex().data(), raw.data(), "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_se3.translation().data(), raw.data() + 2, "");
    SOPHUS_TEST_NEQ(passed, map_of_se3.unitComplex().data(), raw2.data(), "");
    Eigen::Map<SE2Type> shallow_copy = map_of_se3;
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.unitComplex().eval(),
        map_of_se3.unitComplex().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.translation().eval(),
        map_of_se3.translation().eval(),
        "");
    Eigen::Map<SE2Type> const const_map_of_se2 = map_of_se3;
    SOPHUS_TEST_EQUAL(
        passed,
        const_map_of_se2.unitComplex().eval(),
        map_of_se3.unitComplex().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        const_map_of_se2.translation().eval(),
        map_of_se3.translation().eval(),
        "");

    SE2Type const const_se2(
        raw2.template head<2>().eval(), raw2.template tail<2>().eval());
    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_se2.data()[i], raw2.data()[i], "");
    }

    SE2Type se2(raw2.template head<2>().eval(), raw2.template tail<2>().eval());
    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, se2.data()[i], raw2.data()[i], "");
    }

    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, se2.data()[i], raw.data()[i], "");
    }

    SE2Type trans = SE2Type::transX(Scalar(0.2));
    SOPHUS_TEST_APPROX(
        passed, trans.translation().x(), Scalar(0.2), kEpsilon<Scalar>, "");
    trans = SE2Type::transY(Scalar(0.7));
    SOPHUS_TEST_APPROX(
        passed, trans.translation().y(), Scalar(0.7), kEpsilon<Scalar>, "");

    Eigen::Matrix<Scalar, 4, 1> data1;

    Eigen::Matrix<Scalar, 4, 1> data2;
    data1 << Scalar(0), Scalar(1), Scalar(1), Scalar(2);
    data1 << Scalar(1), Scalar(0), Scalar(2), Scalar(1);

    Eigen::Map<SE2Type> map1(data1.data());

    Eigen::Map<SE2Type> map2(data2.data());

    // map -> map assignment
    map2 = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), map2.matrix(), "");

    // map -> type assignment
    SE2Type copy;
    copy = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    // type -> map assignment
    copy = SE2Type::trans(Scalar(4), Scalar(5)) * SE2Type::rot(Scalar(0.5));
    map1 = copy;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    return passed;
  }

  bool testMutatingAccessors() {
    bool passed = true;
    SE2Type se2;
    SO2Type r(Scalar(0.2));
    se2.setRotationMatrix(r.matrix());
    SOPHUS_TEST_APPROX(
        passed, se2.rotationMatrix(), r.matrix(), kEpsilon<Scalar>, "");

    Eigen::Matrix<Scalar, 4, 1> raw;
    raw << Scalar(1), Scalar(0), Scalar(3), Scalar(1);
    Eigen::Map<SE2Type> map_of_se2(raw.data());
    map_of_se2.setRotationMatrix(r.matrix());
    SOPHUS_TEST_APPROX(
        passed, map_of_se2.rotationMatrix(), r.matrix(), kEpsilon<Scalar>, "");

    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Eigen::Matrix3<Scalar> i = Eigen::Matrix3<Scalar>::Identity();
    SOPHUS_TEST_EQUAL(passed, SE2Type().matrix(), i, "");

    SE2Type se2 = se2_vec_.front();
    Point translation = se2.translation();
    SO2Type so2 = se2.so2();

    SOPHUS_TEST_APPROX(
        passed,
        SE2Type(so2.log(), translation).matrix(),
        se2.matrix(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        SE2Type(so2, translation).matrix(),
        se2.matrix(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        SE2Type(so2.matrix(), translation).matrix(),
        se2.matrix(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        SE2Type(so2.unitComplex(), translation).matrix(),
        se2.matrix(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        SE2Type(se2.matrix()).matrix(),
        se2.matrix(),
        kEpsilon<Scalar>,
        "");

    return passed;
  }

  template <class ST = Scalar>
  std::enable_if_t<std::is_floating_point<ST>::value, bool> testFit() {
    bool passed = true;
    for (int i = 0; i < 100; ++i) {
      Eigen::Matrix3<Scalar> t = Eigen::Matrix3<Scalar>::Random();
      SE2Type se2 = SE2Type::fitToSe2(t);
      SE2Type se2_2 = SE2Type::fitToSe2(se2.matrix());

      SOPHUS_TEST_APPROX(
          passed, se2.matrix(), se2_2.matrix(), kEpsilon<Scalar>, "");
    }
    return passed;
  }

  template <class ST = Scalar>
  std::enable_if_t<!std::is_floating_point<ST>::value, bool> testFit() {
    return true;
  }

  std::vector<SE2Type, Eigen::aligned_allocator<SE2Type>> se2_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int testSe2() {
  using std::cerr;
  using std::endl;

  cerr << "Test Se2" << endl << endl;
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

int main() { return sophus::testSe2(); }
