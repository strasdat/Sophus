// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/sim3.h"

#include "sophus/lie/details/test_impl.h"

#include <unsupported/Eigen/MatrixFunctions>

#include <iostream>

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {  // NOLINT
template class Map<sophus::Sim3<double>>;
template class Map<sophus::Sim3<double> const>;
}  // namespace Eigen

namespace sophus {

template class Sim3<double>;
#if SOPHUS_CERES
template class Sim3<ceres::Jet<double, 3>>;
#endif

template <class TScalar>
class Tests {
 public:
  using Scalar = TScalar;
  using Sim3Type = Sim3<Scalar>;
  using RxSo3Type = RxSo3<Scalar>;
  using Point = typename Sim3<Scalar>::Point;
  using Vector4Type = Eigen::Vector4<Scalar>;
  using Tangent = typename Sim3<Scalar>::Tangent;
  Scalar const k_pi = kPi<Scalar>;  // NOLINT

  Tests() {
    sim3_vec_.push_back(Sim3Type(
        RxSo3Type::exp(
            Vector4Type(Scalar(0.2), Scalar(0.5), Scalar(0.0), Scalar(1.))),
        Point(Scalar(0), Scalar(0), Scalar(0))));
    sim3_vec_.push_back(Sim3Type(
        RxSo3Type::exp(
            Vector4Type(Scalar(0.2), Scalar(0.5), Scalar(-1.0), Scalar(1.1))),
        Point(Scalar(10), Scalar(0), Scalar(0))));
    sim3_vec_.push_back(Sim3Type(
        RxSo3Type::exp(
            Vector4Type(Scalar(0.), Scalar(0.), Scalar(0.), Scalar(0.))),
        Point(Scalar(0), Scalar(10), Scalar(5))));
    sim3_vec_.push_back(Sim3Type(
        RxSo3Type::exp(
            Vector4Type(Scalar(0.), Scalar(0.), Scalar(0.), Scalar(1.1))),
        Point(Scalar(0), Scalar(10), Scalar(5))));
    sim3_vec_.push_back(Sim3Type(
        RxSo3Type::exp(
            Vector4Type(Scalar(0.), Scalar(0.), Scalar(0.00001), Scalar(0.))),
        Point(Scalar(0), Scalar(0), Scalar(0))));
    sim3_vec_.push_back(Sim3Type(
        RxSo3Type::exp(Vector4Type(
            Scalar(0.), Scalar(0.), Scalar(0.00001), Scalar(0.0000001))),
        Point(Scalar(1), Scalar(-1.00000001), Scalar(2.0000000001))));
    sim3_vec_.push_back(Sim3Type(
        RxSo3Type::exp(
            Vector4Type(Scalar(0.), Scalar(0.), Scalar(0.00001), Scalar(0))),
        Point(Scalar(0.01), Scalar(0), Scalar(0))));
    sim3_vec_.push_back(Sim3Type(
        RxSo3Type::exp(Vector4Type(k_pi, Scalar(0), Scalar(0), Scalar(0.9))),
        Point(Scalar(4), Scalar(-5), Scalar(0))));
    sim3_vec_.push_back(
        Sim3Type(
            RxSo3Type::exp(
                Vector4Type(Scalar(0.2), Scalar(0.5), Scalar(0.0), Scalar(0))),
            Point(Scalar(0), Scalar(0), Scalar(0))) *
        Sim3Type(
            RxSo3Type::exp(Vector4Type(k_pi, Scalar(0), Scalar(0), Scalar(0))),
            Point(Scalar(0), Scalar(0), Scalar(0))) *
        Sim3Type(
            RxSo3Type::exp(Vector4Type(
                Scalar(-0.2), Scalar(-0.5), Scalar(-0.0), Scalar(0))),
            Point(Scalar(0), Scalar(0), Scalar(0))));
    sim3_vec_.push_back(
        Sim3Type(
            RxSo3Type::exp(
                Vector4Type(Scalar(0.3), Scalar(0.5), Scalar(0.1), Scalar(0))),
            Point(Scalar(2), Scalar(0), Scalar(-7))) *
        Sim3Type(
            RxSo3Type::exp(Vector4Type(k_pi, Scalar(0), Scalar(0), Scalar(0))),
            Point(Scalar(0), Scalar(0), Scalar(0))) *
        Sim3Type(
            RxSo3Type::exp(Vector4Type(
                Scalar(-0.3), Scalar(-0.5), Scalar(-0.1), Scalar(0))),
            Point(Scalar(0), Scalar(6), Scalar(0))));
    Tangent tmp;
    tmp << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
        Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
        Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(1), Scalar(0), Scalar(1), Scalar(0), Scalar(0),
        Scalar(0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(0), Scalar(1), Scalar(0), Scalar(1), Scalar(0),
        Scalar(0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(-1), Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1),
        Scalar(-0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(20), Scalar(-1), Scalar(0), Scalar(-1), Scalar(1), Scalar(0),
        Scalar(-0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(30), Scalar(5), Scalar(-1), Scalar(20), Scalar(-1), Scalar(0),
        Scalar(1.5);
    tangent_vec_.push_back(tmp);

    point_vec_.push_back(Point(Scalar(1), Scalar(2), Scalar(4)));
    point_vec_.push_back(Point(Scalar(1), Scalar(-3), Scalar(0.5)));
    point_vec_.push_back(Point(Scalar(-5), Scalar(-6), Scalar(7)));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testRawDataAcces();
    passed &= testConstructors();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<Sim3Type> tests(sim3_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 7, 1> raw;
    raw << Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(1), Scalar(3),
        Scalar(2);
    Eigen::Map<Sim3Type const> map_of_const_sim3(raw.data());
    SOPHUS_TEST_APPROX(
        passed,
        map_of_const_sim3.quaternion().coeffs().eval(),
        raw.template head<4>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        map_of_const_sim3.translation().eval(),
        raw.template tail<3>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_const_sim3.quaternion().coeffs().data(), raw.data(), "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_const_sim3.translation().data(), raw.data() + 4, "");
    Eigen::Map<Sim3Type const> const_shallow_copy = map_of_const_sim3;
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.quaternion().coeffs().eval(),
        map_of_const_sim3.quaternion().coeffs().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.translation().eval(),
        map_of_const_sim3.translation().eval(),
        "");

    Eigen::Matrix<Scalar, 7, 1> raw2;
    raw2 << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(3), Scalar(2),
        Scalar(1);
    Eigen::Map<Sim3Type> map_of_sim3(raw.data());
    Eigen::Quaternion<Scalar> quat;
    quat.coeffs() = raw2.template head<4>();
    map_of_sim3.setQuaternion(quat);
    map_of_sim3.translation() = raw2.template tail<3>();
    SOPHUS_TEST_APPROX(
        passed,
        map_of_sim3.quaternion().coeffs().eval(),
        raw2.template head<4>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        map_of_sim3.translation().eval(),
        raw2.template tail<3>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_sim3.quaternion().coeffs().data(), raw.data(), "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_sim3.translation().data(), raw.data() + 4, "");
    SOPHUS_TEST_NEQ(
        passed,
        map_of_sim3.quaternion().coeffs().data(),
        quat.coeffs().data(),
        "");
    Eigen::Map<Sim3Type> shallow_copy = map_of_sim3;
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.quaternion().coeffs().eval(),
        map_of_sim3.quaternion().coeffs().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.translation().eval(),
        map_of_sim3.translation().eval(),
        "");
    Eigen::Map<Sim3Type> const const_map_of_sim3 = map_of_sim3;
    SOPHUS_TEST_EQUAL(
        passed,
        const_map_of_sim3.quaternion().coeffs().eval(),
        map_of_sim3.quaternion().coeffs().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        const_map_of_sim3.translation().eval(),
        map_of_sim3.translation().eval(),
        "");

    Sim3Type const const_sim3(quat, raw2.template tail<3>().eval());
    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_sim3.data()[i], raw2.data()[i], "");
    }

    Sim3Type se3(quat, raw2.template tail<3>().eval());
    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, se3.data()[i], raw2.data()[i], "");
    }

    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, se3.data()[i], raw.data()[i], "");
    }

    Eigen::Matrix<Scalar, 7, 1> data1;

    Eigen::Matrix<Scalar, 7, 1> data2;
    data1 << Scalar(0), Scalar(2), Scalar(0), Scalar(0), Scalar(1), Scalar(2),
        Scalar(3);
    data2 << Scalar(0), Scalar(0), Scalar(2), Scalar(0), Scalar(3), Scalar(2),
        Scalar(1);

    Eigen::Map<Sim3Type> map1(data1.data());

    Eigen::Map<Sim3Type> map2(data2.data());

    // map -> map assignment
    map2 = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), map2.matrix(), "");

    // map -> type assignment
    Sim3Type copy;
    copy = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    // type -> map assignment
    copy = Sim3Type(
        RxSo3Type::exp(
            Vector4Type(Scalar(0.2), Scalar(0.5), Scalar(-1.0), Scalar(1.1))),
        Point(Scalar(10), Scalar(0), Scalar(0)));
    map1 = copy;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Eigen::Matrix<Scalar, 4, 4> i = Eigen::Matrix<Scalar, 4, 4>::Identity();
    SOPHUS_TEST_EQUAL(passed, Sim3Type().matrix(), i, "");

    Sim3Type sim3 = sim3_vec_.front();
    Point translation = sim3.translation();
    RxSo3Type rxso3 = sim3.rxso3();

    SOPHUS_TEST_APPROX(
        passed,
        Sim3Type(rxso3, translation).matrix(),
        sim3.matrix(),
        kEpsilon<Scalar>,
        "Sim3(RxSo3, translation)");
    SOPHUS_TEST_APPROX(
        passed,
        Sim3Type(rxso3.quaternion(), translation).matrix(),
        sim3.matrix(),
        kEpsilon<Scalar>,
        "Sim3(quaternion, translation)");
    SOPHUS_TEST_APPROX(
        passed,
        Sim3Type(rxso3.scale(), rxso3.quaternion().normalized(), translation)
            .matrix(),
        sim3.matrix(),
        kEpsilon<Scalar>,
        "Sim3(scale, unit_quaternion, translation)");
    SOPHUS_TEST_APPROX(
        passed,
        Sim3Type(sim3.matrix()).matrix(),
        sim3.matrix(),
        kEpsilon<Scalar>,
        "Sim3(matrix4x4)");

    Scalar scale(1.2);
    sim3.setScale(scale);
    SOPHUS_TEST_APPROX(
        passed, scale, sim3.scale(), kEpsilon<Scalar>, "setScale");

    sim3.setQuaternion(sim3_vec_[0].rxso3().quaternion());
    SOPHUS_TEST_APPROX(
        passed,
        sim3_vec_[0].rxso3().quaternion().coeffs(),
        sim3_vec_[0].rxso3().quaternion().coeffs(),
        kEpsilon<Scalar>,
        "setQuaternion");
    return passed;
  }

  std::vector<Sim3Type, Eigen::aligned_allocator<Sim3Type>> sim3_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int testSim3() {
  using std::cerr;
  using std::endl;

  cerr << "Test Sim3" << endl << endl;
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
