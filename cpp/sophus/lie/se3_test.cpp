// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/se3.h"

#include "sophus/lie/details/test_impl.h"

#include <iostream>

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {  // NOLINT
template class Map<sophus::Se3<double>>;
template class Map<sophus::Se3<double> const>;
}  // namespace Eigen

namespace sophus {

template class Se3<double, Eigen::AutoAlign>;
template class Se3<float, Eigen::DontAlign>;
#if SOPHUS_CERES
template class Se3<ceres::Jet<double, 3>>;
#endif

template <class ScalarT>
class Tests {
 public:
  using Scalar = ScalarT;
  using SE3Type = Se3<Scalar>;
  using SO3Type = So3<Scalar>;
  using Point = typename Se3<Scalar>::Point;
  using Tangent = typename Se3<Scalar>::Tangent;
  Scalar const k_pi = kPi<Scalar>;  // NOLINT

  Tests() {
    se3_vec_ = getTestSE3s<Scalar>();

    Tangent tmp;
    tmp << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(1), Scalar(0), Scalar(1), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(-5), Scalar(10), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(-1), Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(20), Scalar(-1), Scalar(0), Scalar(-1), Scalar(1), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(30), Scalar(5), Scalar(-1), Scalar(20), Scalar(-1), Scalar(0);
    tangent_vec_.push_back(tmp);

    point_vec_.push_back(Point(Scalar(1), Scalar(2), Scalar(4)));
    point_vec_.push_back(Point(Scalar(1), Scalar(-3), Scalar(0.5)));
    point_vec_.push_back(Point(Scalar(-5), Scalar(-6), Scalar(7)));
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
    LieGroupTests<SE3Type> tests(se3_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 7, 1> raw;
    raw << Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(1), Scalar(3),
        Scalar(2);
    Eigen::Map<SE3Type const> map_of_const_se3(raw.data());
    SOPHUS_TEST_APPROX(
        passed,
        map_of_const_se3.unitQuaternion().coeffs().eval(),
        raw.template head<4>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        map_of_const_se3.translation().eval(),
        raw.template tail<3>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        map_of_const_se3.unitQuaternion().coeffs().data(),
        raw.data(),
        "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_const_se3.translation().data(), raw.data() + 4, "");
    Eigen::Map<SE3Type const> const_shallow_copy = map_of_const_se3;
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.unitQuaternion().coeffs().eval(),
        map_of_const_se3.unitQuaternion().coeffs().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.translation().eval(),
        map_of_const_se3.translation().eval(),
        "");

    Eigen::Matrix<Scalar, 7, 1> raw2;
    raw2 << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(3), Scalar(2),
        Scalar(1);
    Eigen::Map<SE3Type> map_of_se3(raw.data());
    Eigen::Quaternion<Scalar> quat;
    quat.coeffs() = raw2.template head<4>();
    map_of_se3.setQuaternion(quat);
    map_of_se3.translation() = raw2.template tail<3>();
    SOPHUS_TEST_APPROX(
        passed,
        map_of_se3.unitQuaternion().coeffs().eval(),
        raw2.template head<4>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        map_of_se3.translation().eval(),
        raw2.template tail<3>().eval(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_se3.unitQuaternion().coeffs().data(), raw.data(), "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_se3.translation().data(), raw.data() + 4, "");
    SOPHUS_TEST_NEQ(
        passed,
        map_of_se3.unitQuaternion().coeffs().data(),
        quat.coeffs().data(),
        "");
    Eigen::Map<SE3Type> shallow_copy = map_of_se3;
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.unitQuaternion().coeffs().eval(),
        map_of_se3.unitQuaternion().coeffs().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.translation().eval(),
        map_of_se3.translation().eval(),
        "");
    Eigen::Map<SE3Type> const const_map_of_se3 = map_of_se3;
    SOPHUS_TEST_EQUAL(
        passed,
        const_map_of_se3.unitQuaternion().coeffs().eval(),
        map_of_se3.unitQuaternion().coeffs().eval(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        const_map_of_se3.translation().eval(),
        map_of_se3.translation().eval(),
        "");

    SE3Type const const_se3(quat, raw2.template tail<3>().eval());
    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_se3.data()[i], raw2.data()[i], "");
    }

    SE3Type se3(quat, raw2.template tail<3>().eval());
    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, se3.data()[i], raw2.data()[i], "");
    }

    for (int i = 0; i < 7; ++i) {
      SOPHUS_TEST_EQUAL(passed, se3.data()[i], raw.data()[i], "");
    }
    SE3Type trans = SE3Type::transX(Scalar(0.2));
    SOPHUS_TEST_APPROX(
        passed, trans.translation().x(), Scalar(0.2), kEpsilon<Scalar>, "");
    trans = SE3Type::transY(Scalar(0.7));
    SOPHUS_TEST_APPROX(
        passed, trans.translation().y(), Scalar(0.7), kEpsilon<Scalar>, "");
    trans = SE3Type::transZ(Scalar(-0.2));
    SOPHUS_TEST_APPROX(
        passed, trans.translation().z(), Scalar(-0.2), kEpsilon<Scalar>, "");
    Tangent t;
    t << Scalar(0), Scalar(0), Scalar(0), Scalar(0.2), Scalar(0), Scalar(0);
    SOPHUS_TEST_EQUAL(
        passed,
        SE3Type::rotX(Scalar(0.2)).matrix(),
        SE3Type::exp(t).matrix(),
        "");
    t << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(-0.2), Scalar(0);
    SOPHUS_TEST_EQUAL(
        passed,
        SE3Type::rotY(Scalar(-0.2)).matrix(),
        SE3Type::exp(t).matrix(),
        "");
    t << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(1.1);
    SOPHUS_TEST_EQUAL(
        passed,
        SE3Type::rotZ(Scalar(1.1)).matrix(),
        SE3Type::exp(t).matrix(),
        "");

    Eigen::Matrix<Scalar, 7, 1> data1;

    Eigen::Matrix<Scalar, 7, 1> data2;
    data1 << Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(1), Scalar(2),
        Scalar(3);
    data1 << Scalar(0), Scalar(0), Scalar(1), Scalar(0), Scalar(3), Scalar(2),
        Scalar(1);

    Eigen::Map<SE3Type> map1(data1.data());

    Eigen::Map<SE3Type> map2(data2.data());

    // map -> map assignment
    map2 = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), map2.matrix(), "");

    // map -> type assignment
    SE3Type copy;
    copy = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    // type -> map assignment
    copy = SE3Type::trans(Scalar(4), Scalar(5), Scalar(6)) *
           SE3Type::rotZ(Scalar(0.5));
    map1 = copy;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    return passed;
  }

  bool testMutatingAccessors() {
    bool passed = true;
    SE3Type se3;
    SO3Type r(SO3Type::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(0.0))));
    se3.setRotationMatrix(r.matrix());
    SOPHUS_TEST_APPROX(
        passed, se3.rotationMatrix(), r.matrix(), kEpsilon<Scalar>, "");

    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Eigen::Matrix<Scalar, 4, 4> i = Eigen::Matrix<Scalar, 4, 4>::Identity();
    SOPHUS_TEST_EQUAL(passed, SE3Type().matrix(), i, "");

    SE3Type se3 = se3_vec_.front();
    Point translation = se3.translation();
    SO3Type so3 = se3.so3();

    SOPHUS_TEST_APPROX(
        passed,
        SE3Type(so3, translation).matrix(),
        se3.matrix(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        SE3Type(so3.matrix(), translation).matrix(),
        se3.matrix(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        SE3Type(so3.unitQuaternion(), translation).matrix(),
        se3.matrix(),
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        SE3Type(se3.matrix()).matrix(),
        se3.matrix(),
        kEpsilon<Scalar>,
        "");

    return passed;
  }

  template <class ST = Scalar>
  std::enable_if_t<std::is_floating_point<ST>::value, bool> testFit() {
    bool passed = true;

    for (int i = 0; i < 100; ++i) {
      Eigen::Matrix4<Scalar> mat_t = Eigen::Matrix4<Scalar>::Random();
      SE3Type se3 = SE3Type::fitToSe3(mat_t);
      SE3Type se3_2 = SE3Type::fitToSe3(se3.matrix());

      SOPHUS_TEST_APPROX(
          passed, se3.matrix(), se3_2.matrix(), kEpsilon<Scalar>, "");
    }
    for (Scalar const angle :
         {Scalar(0.0), Scalar(0.1), Scalar(0.3), Scalar(-0.7)}) {
      SOPHUS_TEST_APPROX(
          passed, SE3Type::rotX(angle).angleX(), angle, kEpsilon<Scalar>, "");
      SOPHUS_TEST_APPROX(
          passed, SE3Type::rotY(angle).angleY(), angle, kEpsilon<Scalar>, "");
      SOPHUS_TEST_APPROX(
          passed, SE3Type::rotZ(angle).angleZ(), angle, kEpsilon<Scalar>, "");
    }
    return passed;
  }

  template <class ST = Scalar>
  std::enable_if_t<!std::is_floating_point<ST>::value, bool> testFit() {
    return true;
  }

  std::vector<SE3Type, Eigen::aligned_allocator<SE3Type>> se3_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int testSe3() {
  using std::cerr;
  using std::endl;

  cerr << "Test Se3" << endl << endl;
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

int main() { return sophus::testSe3(); }
