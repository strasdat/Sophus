// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/rxso3.h"

#include "sophus/lie/details/test_impl.h"

#include <iostream>

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {  // NOLINT
template class Map<sophus::RxSo3<double>>;
template class Map<sophus::RxSo3<double> const>;
}  // namespace Eigen

namespace sophus {

template class RxSo3<double, Eigen::AutoAlign>;
template class RxSo3<float, Eigen::DontAlign>;
#if SOPHUS_CERES
template class RxSo3<ceres::Jet<double, 3>>;
#endif

template <class TScalar>
class Tests {
 public:
  using Scalar = TScalar;
  using So3Type = So3<Scalar>;
  using RxSo3Type = RxSo3<Scalar>;
  using RotationMatrixType = typename So3<Scalar>::Transformation;
  using Point = typename RxSo3<Scalar>::Point;
  using Tangent = typename RxSo3<Scalar>::Tangent;
  Scalar const k_pi = kPi<Scalar>;  // NOLINT

  Tests() {
    rxso3_vec_.push_back(RxSo3Type::exp(
        Tangent(Scalar(0.2), Scalar(0.5), Scalar(0.0), Scalar(1.))));
    rxso3_vec_.push_back(RxSo3Type::exp(
        Tangent(Scalar(0.2), Scalar(0.5), Scalar(-1.0), Scalar(1.1))));
    rxso3_vec_.push_back(RxSo3Type::exp(
        Tangent(Scalar(0.), Scalar(0.), Scalar(0.), Scalar(1.1))));
    rxso3_vec_.push_back(RxSo3Type::exp(
        Tangent(Scalar(0.), Scalar(0.), Scalar(0.00001), Scalar(0.))));
    rxso3_vec_.push_back(RxSo3Type::exp(
        Tangent(Scalar(0.), Scalar(0.), Scalar(0.00001), Scalar(0.00001))));
    rxso3_vec_.push_back(RxSo3Type::exp(
        Tangent(Scalar(0.), Scalar(0.), Scalar(0.00001), Scalar(0))));
    rxso3_vec_.push_back(
        RxSo3Type::exp(Tangent(k_pi, Scalar(0), Scalar(0), Scalar(0.9))));
    rxso3_vec_.push_back(
        RxSo3Type::exp(
            Tangent(Scalar(0.2), Scalar(-0.5), Scalar(0), Scalar(0))) *
        RxSo3Type::exp(Tangent(k_pi, Scalar(0), Scalar(0), Scalar(0))) *
        RxSo3Type::exp(
            Tangent(-Scalar(0.2), Scalar(-0.5), Scalar(0), Scalar(0))));
    rxso3_vec_.push_back(
        RxSo3Type::exp(
            Tangent(Scalar(0.3), Scalar(0.5), Scalar(0.1), Scalar(0))) *
        RxSo3Type::exp(Tangent(k_pi, Scalar(0), Scalar(0), Scalar(0))) *
        RxSo3Type::exp(
            Tangent(Scalar(-0.3), Scalar(-0.5), Scalar(-0.1), Scalar(0))));

    Tangent tmp;
    tmp << Scalar(0), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0), Scalar(0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(1), Scalar(0), Scalar(0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(0), Scalar(1), Scalar(-0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(-1), Scalar(1), Scalar(0), Scalar(-0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(20), Scalar(-1), Scalar(0), Scalar(2);
    tangent_vec_.push_back(tmp);

    point_vec_.push_back(Point(Scalar(1), Scalar(2), Scalar(4)));
    point_vec_.push_back(Point(Scalar(1), Scalar(-3), Scalar(0.5)));
    point_vec_.push_back(Point(Scalar(-5), Scalar(-6), Scalar(7)));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testSaturation();
    passed &= testRawDataAcces();
    passed &= testConstructors();
    passed &= testFit();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<RxSo3Type> tests(rxso3_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testSaturation() {
    using std::log;

    bool passed = true;
    // Test if product of two small group elements has correct scale
    RxSo3Type small1(kEpsilon<Scalar>, So3Type());
    RxSo3Type small2(
        kEpsilon<Scalar>,
        So3Type::exp(
            Eigen::Vector3<Scalar>(kPi<Scalar>, Scalar(0), Scalar(0))));
    RxSo3Type saturated_product = small1 * small2;
    SOPHUS_TEST_APPROX(
        passed,
        saturated_product.scale(),
        kEpsilon<Scalar>,
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        saturated_product.so3().matrix(),
        (small1.so3() * small2.so3()).matrix(),
        kEpsilon<Scalar>,
        "");
    /*
     * Test if group exponential produces group elements
     * that can be multiplied safely even for large scale factors
     */
    const Tangent large_log(
        Scalar(1.), Scalar(2.), Scalar(3.), std::numeric_limits<Scalar>::max());
    const Tangent regular_log(Scalar(4.), Scalar(5.), Scalar(6.), Scalar(0.));
    const RxSo3Type large = RxSo3Type::exp(large_log);
    const RxSo3Type regular = RxSo3Type::exp(regular_log);
    const RxSo3Type product = regular * large;
    SOPHUS_TEST(passed, isfinite(large.scale()), "");
    SOPHUS_TEST(passed, isfinite(product.scale()), "");

    // Test if saturation is handled correctly with imprecision of IEEE754-2008
    Tangent small_log;
    while (true) {
      // Note: use cast() since Random() doesn't work with ceres::Jet Scalar
      // types
      const typename So3Type::Tangent so3_tangent =
          Eigen::Vector3d::Random().cast<Scalar>();
      const So3Type so3_exp = So3Type::exp(so3_tangent);
      if (so3_exp.unitQuaternion().squaredNorm() >= Scalar(1.)) {
        continue;
      }
      small_log << so3_tangent, log(kEpsilon<Scalar> / Scalar(2.));
      break;
    }

    const RxSo3Type small_exp = RxSo3Type::exp(small_log);
    SOPHUS_TEST_APPROX(
        passed,
        small_exp.quaternion().squaredNorm(),
        kEpsilon<Scalar>,
        kEpsilon<Scalar>,
        "");
    return passed;
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 4, 1> raw = {
        Scalar(0), Scalar(1), Scalar(0), Scalar(0)};
    Eigen::Map<RxSo3Type const> map_of_const_rxso3(raw.data());
    SOPHUS_TEST_APPROX(
        passed,
        map_of_const_rxso3.quaternion().coeffs().eval(),
        raw,
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        map_of_const_rxso3.quaternion().coeffs().data(),
        raw.data(),
        "");
    Eigen::Map<RxSo3Type const> const_shallow_copy = map_of_const_rxso3;
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.quaternion().coeffs().eval(),
        map_of_const_rxso3.quaternion().coeffs().eval(),
        "");

    Eigen::Matrix<Scalar, 4, 1> raw2 = {
        Scalar(1), Scalar(0), Scalar(0), Scalar(0)};
    Eigen::Map<RxSo3Type> map_of_rxso3(raw.data());
    Eigen::Quaternion<Scalar> quat;
    quat.coeffs() = raw2;
    map_of_rxso3.setQuaternion(quat);
    SOPHUS_TEST_APPROX(
        passed,
        map_of_rxso3.quaternion().coeffs().eval(),
        raw2,
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_rxso3.quaternion().coeffs().data(), raw.data(), "");
    SOPHUS_TEST_NEQ(
        passed,
        map_of_rxso3.quaternion().coeffs().data(),
        quat.coeffs().data(),
        "");
    Eigen::Map<RxSo3Type> shallow_copy = map_of_rxso3;
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.quaternion().coeffs().eval(),
        map_of_rxso3.quaternion().coeffs().eval(),
        "");

    RxSo3Type const const_so3(quat);
    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_so3.data()[i], raw2.data()[i], "");
    }

    RxSo3Type so3(quat);
    for (int i = 0; i < 4; ++i) {
      so3.data()[i] = raw[i];
    }

    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, so3.data()[i], raw.data()[i], "");
    }

    // regression: test that rotationMatrix API doesn't change underlying value
    // for non-const-map and compiles at all for const-map
    Eigen::Matrix<Scalar, 4, 1> raw3 = {
        Scalar(2), Scalar(0), Scalar(0), Scalar(0)};
    Eigen::Map<RxSo3Type> map_of_rxso3_3(raw3.data());
    Eigen::Map<const RxSo3Type> const_map_of_rxso3_3(raw3.data());
    RxSo3Type rxso3_copy3 = map_of_rxso3_3;
    const RotationMatrixType r_ref = map_of_rxso3_3.so3().matrix();

    const RotationMatrixType r = map_of_rxso3_3.rotationMatrix();
    SOPHUS_TEST_APPROX(passed, r_ref, r, kEpsilon<Scalar>, "");
    SOPHUS_TEST_APPROX(
        passed,
        map_of_rxso3_3.quaternion().coeffs().eval(),
        rxso3_copy3.quaternion().coeffs().eval(),
        kEpsilon<Scalar>,
        "");

    const RotationMatrixType r_const = const_map_of_rxso3_3.rotationMatrix();
    SOPHUS_TEST_APPROX(passed, r_ref, r_const, kEpsilon<Scalar>, "");
    SOPHUS_TEST_APPROX(
        passed,
        const_map_of_rxso3_3.quaternion().coeffs().eval(),
        rxso3_copy3.quaternion().coeffs().eval(),
        kEpsilon<Scalar>,
        "");

    Eigen::Matrix<Scalar, 4, 1> data1;

    Eigen::Matrix<Scalar, 4, 1> data2;
    data1 << Scalar(.1), Scalar(.2), Scalar(.3), Scalar(.4);
    data2 << Scalar(.5), Scalar(.4), Scalar(.3), Scalar(.2);

    Eigen::Map<RxSo3Type> map1(data1.data());

    Eigen::Map<RxSo3Type> map2(data2.data());

    // map -> map assignment
    map2 = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), map2.matrix(), "");

    // map -> type assignment
    RxSo3Type copy;
    copy = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    // type -> map assignment
    copy = RxSo3Type::exp(
        Tangent(Scalar(0.2), Scalar(0.5), Scalar(-1.0), Scalar(1.1)));
    map1 = copy;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    RxSo3Type rxso3;
    Scalar scale(1.2);
    rxso3.setScale(scale);
    SOPHUS_TEST_APPROX(
        passed, scale, rxso3.scale(), kEpsilon<Scalar>, "setScale");
    auto so3 = rxso3_vec_[0].so3();
    rxso3.setSO3(so3);
    SOPHUS_TEST_APPROX(
        passed, scale, rxso3.scale(), kEpsilon<Scalar>, "setScale");
    SOPHUS_TEST_APPROX(
        passed,
        RxSo3Type(scale, so3).matrix(),
        rxso3.matrix(),
        kEpsilon<Scalar>,
        "RxSo3(scale, So3)");
    SOPHUS_TEST_APPROX(
        passed,
        RxSo3Type(scale, so3.matrix()).matrix(),
        rxso3.matrix(),
        kEpsilon<Scalar>,
        "RxSo3(scale, matrix3x3)");
    const Eigen::Quaternion<Scalar> q = rxso3.quaternion();
    SOPHUS_TEST_APPROX(
        passed,
        RxSo3Type(scale, q).matrix(),
        RxSo3Type(scale, So3<Scalar>(q)).matrix(),
        kEpsilon<Scalar>,
        "RxSo3(scale, unit_quaternion)");
    Eigen::Matrix3<Scalar> r =
        So3<Scalar>::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(-1.0)))
            .matrix();
    Eigen::Matrix3<Scalar> s_r = r * Scalar(1.3);
    SOPHUS_TEST_APPROX(
        passed, RxSo3Type(s_r).matrix(), s_r, kEpsilon<Scalar>, "RxSo3(sR)");
    rxso3.setScaledRotationMatrix(s_r);
    SOPHUS_TEST_APPROX(
        passed,
        s_r,
        rxso3.matrix(),
        kEpsilon<Scalar>,
        "setScaleRotationMatrix");
    rxso3.setScale(scale);
    rxso3.setRotationMatrix(r);
    SOPHUS_TEST_APPROX(
        passed,
        r,
        rxso3.rotationMatrix(),
        kEpsilon<Scalar>,
        "setRotationMatrix");
    SOPHUS_TEST_APPROX(
        passed, scale, rxso3.scale(), kEpsilon<Scalar>, "setScale");

    return passed;
  }

  template <class TS = Scalar>
  std::enable_if_t<std::is_floating_point<TS>::value, bool> testFit() {
    bool passed = true;
    for (int i = 0; i < 10; ++i) {
      Eigen::Matrix3<Scalar> m = Eigen::Matrix3<Scalar>::Random();
      for (Scalar scale : {Scalar(0.01), Scalar(0.99), Scalar(1), Scalar(10)}) {
        Eigen::Matrix3<Scalar> r = makeRotationMatrix(m);
        Eigen::Matrix3<Scalar> s_r = scale * r;
        SOPHUS_TEST(
            passed,
            isScaledOrthogonalAndPositive(s_r),
            "isScaledOrthogonalAndPositive(sR): {} *\n{}",
            scale,
            r);
        Eigen::Matrix3<Scalar> s_r_cols_swapped;
        s_r_cols_swapped << s_r.col(1), s_r.col(0), s_r.col(2);
        SOPHUS_TEST(
            passed,
            !isScaledOrthogonalAndPositive(s_r_cols_swapped),
            "isScaledOrthogonalAndPositive(-sR): {} *\n{}",
            scale,
            r);
      }
    }
    return passed;
  }

  template <class TS = Scalar>
  std::enable_if_t<!std::is_floating_point<TS>::value, bool> testFit() {
    return true;
  }

  std::vector<RxSo3Type, Eigen::aligned_allocator<RxSo3Type>> rxso3_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int testRxso3() {
  using std::cerr;
  using std::endl;

  cerr << "Test RxSo3" << endl << endl;
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

int main() { return sophus::testRxso3(); }
