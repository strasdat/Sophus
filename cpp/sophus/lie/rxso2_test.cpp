// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/rxso2.h"

#include "sophus/lie/details/test_impl.h"

#include <iostream>

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {  // NOLINT
template class Map<sophus::RxSo2<double>>;
template class Map<sophus::RxSo2<double> const>;
}  // namespace Eigen

namespace sophus {

template class RxSo2<double>;
#if SOPHUS_CERES
template class RxSo2<ceres::Jet<double, 3>>;
#endif

template <class TScalar>
class Tests {
 public:
  using Scalar = TScalar;
  using So2Type = So2<Scalar>;
  using RxSo2Type = RxSo2<Scalar>;
  using RotationMatrixType = typename So2<Scalar>::Transformation;
  using Point = typename RxSo2<Scalar>::Point;
  using Tangent = typename RxSo2<Scalar>::Tangent;
  Scalar const k_pi = kPi<Scalar>;  // NOLINT

  Tests() {
    rxso2_vec_.push_back(RxSo2Type::exp(Tangent(0.2, 1.)));
    rxso2_vec_.push_back(RxSo2Type::exp(Tangent(0.2, 1.1)));
    rxso2_vec_.push_back(RxSo2Type::exp(Tangent(0., 1.1)));
    rxso2_vec_.push_back(RxSo2Type::exp(Tangent(0.00001, 0.)));
    rxso2_vec_.push_back(RxSo2Type::exp(Tangent(0.00001, 0.00001)));
    rxso2_vec_.push_back(RxSo2Type::exp(Tangent(k_pi, 0.9)));
    rxso2_vec_.push_back(
        RxSo2Type::exp(Tangent(0.2, 0)) * RxSo2Type::exp(Tangent(k_pi, 0.0)) *
        RxSo2Type::exp(Tangent(-0.2, 0)));
    rxso2_vec_.push_back(
        RxSo2Type::exp(Tangent(0.3, 0)) * RxSo2Type::exp(Tangent(k_pi, 0.001)) *
        RxSo2Type::exp(Tangent(-0.3, 0)));

    Tangent tmp;
    tmp << Scalar(0), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(1), Scalar(0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(0), Scalar(-0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(-1), Scalar(-0.1);
    tangent_vec_.push_back(tmp);
    tmp << Scalar(20), Scalar(2);
    tangent_vec_.push_back(tmp);

    point_vec_.push_back(Point(Scalar(1), Scalar(4)));
    point_vec_.push_back(Point(Scalar(1), Scalar(-3)));
  }

  template <class TS = Scalar>
  std::enable_if_t<std::is_floating_point<TS>::value, bool> testFit() {
    bool passed = true;
    for (int i = 0; i < 10; ++i) {
      Eigen::Matrix2<Scalar> m = Eigen::Matrix2<Scalar>::Random();
      for (Scalar scale : {Scalar(0.01), Scalar(0.99), Scalar(1), Scalar(10)}) {
        Eigen::Matrix2<Scalar> r = makeRotationMatrix(m);
        Eigen::Matrix2<Scalar> s_r = scale * r;
        SOPHUS_TEST(
            passed,
            isScaledOrthogonalAndPositive(s_r),
            "isScaledOrthogonalAndPositive(sR): {} *\n{}",
            scale,
            r);
        Eigen::Matrix2<Scalar> s_r_cols_swapped;
        s_r_cols_swapped << s_r.col(1), s_r.col(0);
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
    LieGroupTests<RxSo2Type> tests(rxso2_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testSaturation() {
    using std::cos;
    using std::log;
    using std::sin;

    bool passed = true;
    // Test if product of two small group elements has correct scale
    RxSo2Type small1(Scalar(1.1) * kEpsilon<Scalar>, So2Type());
    RxSo2Type small2(Scalar(1.1) * kEpsilon<Scalar>, So2Type::exp(kPi<Scalar>));
    RxSo2Type saturated_product = small1 * small2;
    SOPHUS_TEST_APPROX(
        passed,
        saturated_product.scale(),
        kEpsilon<Scalar>,
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_APPROX(
        passed,
        saturated_product.so2().matrix(),
        (small1.so2() * small2.so2()).matrix(),
        kEpsilon<Scalar>,
        "");

    /*
     * Test if group exponential produces group elements
     * that can be multiplied safely even for large scale factors
     */
    const Tangent large_log(Scalar(1.), std::numeric_limits<Scalar>::max());
    const Tangent regular_log(Scalar(2.), Scalar(0.));
    const RxSo2Type large = RxSo2Type::exp(large_log);
    const RxSo2Type regular = RxSo2Type::exp(regular_log);
    const RxSo2Type product = regular * large;
    SOPHUS_TEST(passed, isfinite(large.scale()), "");
    SOPHUS_TEST(passed, isfinite(product.scale()), "");

    // Test if saturation is handled correctly with imprecision of IEEE754-2008
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform(0., kPi<double>);
    Tangent small_log;
    while (true) {
      // Note: sample double and convert to Scalar for compatibility with
      // ceres::Jet
      const Scalar phi = Scalar(uniform(rng));
      const Scalar c = cos(phi);
      const Scalar s = sin(phi);
      if (c * c + s * s < Scalar(1.)) {
        small_log[0] = phi;
        break;
      }
    }
    small_log[1] = log(kEpsilon<Scalar> / Scalar(2.));

    const RxSo2Type small_exp = RxSo2Type::exp(small_log);
    SOPHUS_TEST_APPROX(
        passed, small_exp.scale(), kEpsilon<Scalar>, kEpsilon<Scalar>, "");
    return passed;
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 2, 1> raw = {0, 1};
    Eigen::Map<RxSo2Type const> map_of_const_rxso2(raw.data());
    SOPHUS_TEST_APPROX(
        passed, map_of_const_rxso2.complex().eval(), raw, kEpsilon<Scalar>, "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_const_rxso2.complex().data(), raw.data(), "");
    Eigen::Map<RxSo2Type const> const_shallow_copy = map_of_const_rxso2;
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.complex().eval(),
        map_of_const_rxso2.complex().eval(),
        "");

    Eigen::Matrix<Scalar, 2, 1> raw2{1, 0};
    Eigen::Map<RxSo2Type> map_of_rxso2(raw2.data());
    SOPHUS_TEST_APPROX(
        passed, map_of_rxso2.complex().eval(), raw2, kEpsilon<Scalar>, "");
    SOPHUS_TEST_EQUAL(passed, map_of_rxso2.complex().data(), raw2.data(), "");
    Eigen::Map<RxSo2Type> shallow_copy = map_of_rxso2;
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.complex().eval(),
        map_of_rxso2.complex().eval(),
        "");

    RxSo2Type const const_so2(raw2);
    for (int i = 0; i < 2; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_so2.data()[i], raw2.data()[i], "");
    }

    RxSo2Type so2(raw2);
    for (int i = 0; i < 2; ++i) {
      so2.data()[i] = raw[i];
    }

    for (int i = 0; i < 2; ++i) {
      SOPHUS_TEST_EQUAL(passed, so2.data()[i], raw.data()[i], "");
    }

    // regression: test that rotationMatrix API doesn't change underlying value
    // for non-const-map and compiles at all for const-map
    Eigen::Matrix<Scalar, 2, 1> raw3 = {Scalar(2), Scalar(0)};
    Eigen::Map<RxSo2Type> map_of_rxso2_3(raw3.data());
    Eigen::Map<const RxSo2Type> const_map_of_rxso2_3(raw3.data());
    RxSo2Type rxso2_copy3 = map_of_rxso2_3;
    const RotationMatrixType r_ref = map_of_rxso2_3.so2().matrix();

    const RotationMatrixType r = map_of_rxso2_3.rotationMatrix();
    SOPHUS_TEST_APPROX(passed, r_ref, r, kEpsilon<Scalar>, "");
    SOPHUS_TEST_APPROX(
        passed,
        map_of_rxso2_3.complex().eval(),
        rxso2_copy3.complex().eval(),
        kEpsilon<Scalar>,
        "");

    const RotationMatrixType r_const = const_map_of_rxso2_3.rotationMatrix();
    SOPHUS_TEST_APPROX(passed, r_ref, r_const, kEpsilon<Scalar>, "");
    SOPHUS_TEST_APPROX(
        passed,
        const_map_of_rxso2_3.complex().eval(),
        rxso2_copy3.complex().eval(),
        kEpsilon<Scalar>,
        "");

    Eigen::Matrix<Scalar, 2, 1> data1;

    Eigen::Matrix<Scalar, 2, 1> data2;
    data1 << Scalar(.1), Scalar(.2);
    data2 << Scalar(.5), Scalar(.4);

    Eigen::Map<RxSo2Type> map1(data1.data());

    Eigen::Map<RxSo2Type> map2(data2.data());

    // map -> map assignment
    map2 = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), map2.matrix(), "");

    // map -> type assignment
    RxSo2Type copy;
    copy = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    // type -> map assignment
    copy = RxSo2Type::exp(Tangent(Scalar(0.2), Scalar(0.5)));
    map1 = copy;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    RxSo2Type rxso2;
    Scalar scale(1.2);
    rxso2.setScale(scale);
    SOPHUS_TEST_APPROX(
        passed, scale, rxso2.scale(), kEpsilon<Scalar>, "setScale");
    Scalar angle(0.2);
    rxso2.setAngle(angle);
    SOPHUS_TEST_APPROX(
        passed, angle, rxso2.angle(), kEpsilon<Scalar>, "setAngle");
    SOPHUS_TEST_APPROX(
        passed,
        scale,
        rxso2.scale(),
        kEpsilon<Scalar>,
        "setAngle leaves scale as is");

    auto so2 = rxso2_vec_[0].so2();
    rxso2.setSO2(so2);
    SOPHUS_TEST_APPROX(
        passed, scale, rxso2.scale(), kEpsilon<Scalar>, "setSO2");
    SOPHUS_TEST_APPROX(
        passed,
        RxSo2Type(scale, so2).matrix(),
        rxso2.matrix(),
        kEpsilon<Scalar>,
        "RxSo2(scale, So2)");
    SOPHUS_TEST_APPROX(
        passed,
        RxSo2Type(scale, so2.matrix()).matrix(),
        rxso2.matrix(),
        kEpsilon<Scalar>,
        "RxSo2(scale, So2)");
    Eigen::Matrix2<Scalar> r = So2<Scalar>::exp(Scalar(0.2)).matrix();
    Eigen::Matrix2<Scalar> s_r = r * Scalar(1.3);
    SOPHUS_TEST_APPROX(
        passed, RxSo2Type(s_r).matrix(), s_r, kEpsilon<Scalar>, "RxSo2(sR)");
    rxso2.setScaledRotationMatrix(s_r);
    SOPHUS_TEST_APPROX(
        passed,
        s_r,
        rxso2.matrix(),
        kEpsilon<Scalar>,
        "setScaleRotationMatrix");
    rxso2.setScale(scale);
    rxso2.setRotationMatrix(r);
    SOPHUS_TEST_APPROX(
        passed,
        r,
        rxso2.rotationMatrix(),
        kEpsilon<Scalar>,
        "setRotationMatrix");
    SOPHUS_TEST_APPROX(
        passed, scale, rxso2.scale(), kEpsilon<Scalar>, "setScale");

    return passed;
  }

  std::vector<RxSo2Type, Eigen::aligned_allocator<RxSo2Type>> rxso2_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int testRxso2() {
  using std::cerr;
  using std::endl;

  cerr << "Test RxSo2" << endl << endl;
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

int main() { return sophus::testRxso2(); }
