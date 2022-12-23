// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/sim2.h"

#include "sophus/lie/details/test_impl.h"

#include <unsupported/Eigen/MatrixFunctions>

#include <iostream>

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {  // NOLINT
template class Map<sophus::Sim2<double>>;
template class Map<sophus::Sim2<double> const>;
}  // namespace Eigen

namespace sophus {

template class Sim2<double>;
#if SOPHUS_CERES
template class Sim2<ceres::Jet<double, 3>>;
#endif

template <class TScalar>
class Tests {
 public:
  using Scalar = TScalar;

  using Sim2Type = Sim2<Scalar>;
  using RxSo2Type = RxSo2<Scalar>;
  using Point = typename Sim2<Scalar>::Point;
  using Vector2Type = Eigen::Vector2<Scalar>;
  using Tangent = typename Sim2<Scalar>::Tangent;
  Scalar const k_pi = kPi<Scalar>;  // NOLINT

  Tests() {
    sim2_vec_.push_back(
        Sim2Type(RxSo2Type::exp(Vector2Type(0.2, 1.)), Point(0, 0)));
    sim2_vec_.push_back(
        Sim2Type(RxSo2Type::exp(Vector2Type(0.2, 1.1)), Point(10, 0)));
    sim2_vec_.push_back(
        Sim2Type(RxSo2Type::exp(Vector2Type(0., 0.)), Point(0, 10)));
    sim2_vec_.push_back(
        Sim2Type(RxSo2Type::exp(Vector2Type(0.00001, 0.)), Point(0, 0)));
    sim2_vec_.push_back(Sim2Type(
        RxSo2Type::exp(Vector2Type(0.00001, 0.0000001)),
        Point(1, -1.00000001)));
    sim2_vec_.push_back(
        Sim2Type(RxSo2Type::exp(Vector2Type(0., 0.)), Point(0.01, 0)));
    sim2_vec_.push_back(
        Sim2Type(RxSo2Type::exp(Vector2Type(k_pi, 0.9)), Point(4, 0)));
    sim2_vec_.push_back(
        Sim2Type(RxSo2Type::exp(Vector2Type(0.2, 0)), Point(0, 0)) *
        Sim2Type(RxSo2Type::exp(Vector2Type(k_pi, 0)), Point(0, 0)) *
        Sim2Type(RxSo2Type::exp(Vector2Type(-0.2, 0)), Point(0, 0)));
    sim2_vec_.push_back(
        Sim2Type(RxSo2Type::exp(Vector2Type(0.3, 0)), Point(2, -7)) *
        Sim2Type(RxSo2Type::exp(Vector2Type(k_pi, 0)), Point(0, 0)) *
        Sim2Type(RxSo2Type::exp(Vector2Type(-0.3, 0)), Point(0, 6)));
    Tangent tmp;
    tmp << Scalar(0), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(1), Scalar(0), Scalar(0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(-1), Scalar(1), Scalar(1), Scalar(-0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(20), Scalar(-1), Scalar(0), Scalar(-0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(30), Scalar(5), Scalar(-1), Scalar(1.5);
    tangent_vec_.push_back(tmp);

    point_vec_.push_back(Point(Scalar(1), Scalar(4)));
    point_vec_.push_back(Point(Scalar(1), Scalar(-3)));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testRawDataAcces();
    passed &= testConstructors();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<Sim2Type> tests(sim2_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 4, 1> raw;
    raw << Scalar(0), Scalar(1), Scalar(3), Scalar(2);
    Eigen::Map<Sim2Type const> map_of_const_sim2(raw.data());
    SOPHUS_TEST_APPROX(
        passed,
        map_of_const_sim2.complex().eval(),
        raw.template head<2>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        map_of_const_sim2.translation().eval(),
        raw.template tail<2>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_const_sim2.complex().data(), raw.data(), "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_const_sim2.translation().data(), raw.data() + 2, "");
    Eigen::Map<Sim2Type const> const_shallow_copy = map_of_const_sim2;
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.complex().eval(),
        map_of_const_sim2.complex().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.translation().eval(),
        map_of_const_sim2.translation().eval(),
        "");

    Eigen::Matrix<Scalar, 4, 1> raw2;
    raw2 << Scalar(1), Scalar(0), Scalar(2), Scalar(1);
    Eigen::Map<Sim2Type> map_of_sim2(raw.data());
    Eigen::Vector2<Scalar> z;
    z = raw2.template head<2>();
    map_of_sim2.setComplex(z);
    map_of_sim2.translation() = raw2.template tail<2>();
    SOPHUS_TEST_APPROX(
        passed,
        map_of_sim2.complex().eval(),
        raw2.template head<2>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        map_of_sim2.translation().eval(),
        raw2.template tail<2>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(passed, map_of_sim2.complex().data(), raw.data(), "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_sim2.translation().data(), raw.data() + 2, "");
    SOPHUS_TEST_NEQ(passed, map_of_sim2.complex().data(), z.data(), "");
    Eigen::Map<Sim2Type> shallow_copy = map_of_sim2;
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.complex().eval(),
        map_of_sim2.complex().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.translation().eval(),
        map_of_sim2.translation().eval(),
        "");
    Eigen::Map<Sim2Type> const const_map_of_sim3 = map_of_sim2;
    SOPHUS_TEST_EQUAL(
        passed,
        const_map_of_sim3.complex().eval(),
        map_of_sim2.complex().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        const_map_of_sim3.translation().eval(),
        map_of_sim2.translation().eval(),
        "");

    Sim2Type const const_sim2(z, raw2.template tail<2>().eval());
    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_sim2.data()[i], raw2.data()[i], "");
    }

    Sim2Type se3(z, raw2.template tail<2>().eval());
    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, se3.data()[i], raw2.data()[i], "");
    }

    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, se3.data()[i], raw.data()[i], "");
    }

    Eigen::Matrix<Scalar, 4, 1> data1;

    Eigen::Matrix<Scalar, 4, 1> data2;
    data1 << Scalar(0), Scalar(2), Scalar(1), Scalar(2);
    data2 << Scalar(2), Scalar(0), Scalar(2), Scalar(1);

    Eigen::Map<Sim2Type> map1(data1.data());

    Eigen::Map<Sim2Type> map2(data2.data());

    // map -> map assignment
    map2 = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), map2.matrix(), "");

    // map -> type assignment
    Sim2Type copy;
    copy = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    // type -> map assignment
    copy = Sim2Type(
        RxSo2Type::exp(Vector2Type(-1, 1)), Point(Scalar(10), Scalar(0)));
    map1 = copy;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Eigen::Matrix<Scalar, 3, 3> i = Eigen::Matrix<Scalar, 3, 3>::Identity();
    SOPHUS_TEST_EQUAL(passed, Sim2Type().matrix(), i, "");

    Sim2Type sim2 = sim2_vec_.front();
    Point translation = sim2.translation();
    RxSo2Type rxso2 = sim2.rxso2();

    SOPHUS_TEST_APPROX(
        passed,
        Sim2Type(rxso2, translation).matrix(),
        sim2.matrix(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        Sim2Type(rxso2.complex(), translation).matrix(),
        sim2.matrix(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        Sim2Type(sim2.matrix()).matrix(),
        sim2.matrix(),
        kEpsilon<Scalar>,
        "");

    Scalar scale(1.2);
    sim2.setScale(scale);
    SOPHUS_TEST_APPROX(
        passed, scale, sim2.scale(), kEpsilon<Scalar>, "setScale", "");

    sim2.setComplex(sim2_vec_[0].rxso2().complex());
    SOPHUS_TEST_APPROX(
        passed,
        sim2_vec_[0].rxso2().complex(),
        sim2_vec_[0].rxso2().complex(),
        kEpsilon<Scalar>,
        "setComplex",
        "");
    return passed;
  }

  std::vector<Sim2Type, Eigen::aligned_allocator<Sim2Type>> sim2_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int testSim3() {
  using std::cerr;
  using std::endl;

  cerr << "Test Sim2" << endl << endl;
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

int main() { return sophus::testSim3(); }
