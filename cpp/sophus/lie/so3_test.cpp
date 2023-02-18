// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/so3.h"

#include "sophus/lie/details/test_impl.h"
#include "sophus/lie/interp/interpolate.h"

#include <iostream>

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {  // NOLINT
template class Map<sophus::So3<double>>;
template class Map<sophus::So3<double> const>;
}  // namespace Eigen

namespace sophus {

template class So3<double>;
#if SOPHUS_CERES
template class So3<ceres::Jet<double, 3>>;
#endif
}  // namespace sophus

namespace sophus {

template <class TScalar>
class Tests {
 public:
  using Scalar = TScalar;
  using So3Type = So3<Scalar>;
  using Point = typename So3<Scalar>::Point;
  using Tangent = typename So3<Scalar>::Tangent;
  Scalar const k_pi = kPi<Scalar>;  // NOLINT

  Tests() {
    so3_vec_.push_back(So3Type(Eigen::Quaternion<Scalar>(
        Scalar(0.1e-11), Scalar(0.), Scalar(1.), Scalar(0.))));
    so3_vec_.push_back(So3Type(Eigen::Quaternion<Scalar>(
        Scalar(-1), Scalar(0.00001), Scalar(0.0), Scalar(0.0))));
    so3_vec_.push_back(
        So3Type::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(0.0))));
    so3_vec_.push_back(
        So3Type::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(-1.0))));
    so3_vec_.push_back(So3Type::exp(Point(Scalar(0.), Scalar(0.), Scalar(0.))));
    so3_vec_.push_back(
        So3Type::exp(Point(Scalar(0.), Scalar(0.), Scalar(0.00001))));
    so3_vec_.push_back(So3Type::exp(Point(k_pi, Scalar(0), Scalar(0))));
    so3_vec_.push_back(
        So3Type::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(0.0))) *
        So3Type::exp(Point(k_pi, Scalar(0), Scalar(0))) *
        So3Type::exp(Point(Scalar(-0.2), Scalar(-0.5), Scalar(-0.0))));
    so3_vec_.push_back(
        So3Type::exp(Point(Scalar(0.3), Scalar(0.5), Scalar(0.1))) *
        So3Type::exp(Point(k_pi, Scalar(0), Scalar(0))) *
        So3Type::exp(Point(Scalar(-0.3), Scalar(-0.5), Scalar(-0.1))));
    tangent_vec_.push_back(Tangent(Scalar(0), Scalar(0), Scalar(0)));
    tangent_vec_.push_back(Tangent(Scalar(1), Scalar(0), Scalar(0)));
    tangent_vec_.push_back(Tangent(Scalar(0), Scalar(1), Scalar(0)));
    tangent_vec_.push_back(
        Tangent(Scalar(k_pi / 2.), Scalar(k_pi / 2.), Scalar(0)));
    tangent_vec_.push_back(Tangent(Scalar(-1), Scalar(1), Scalar(0)));
    tangent_vec_.push_back(Tangent(Scalar(20), Scalar(-1), Scalar(0)));
    tangent_vec_.push_back(Tangent(Scalar(30), Scalar(5), Scalar(-1)));

    point_vec_.push_back(Point(Scalar(1), Scalar(2), Scalar(4)));
    point_vec_.push_back(Point(Scalar(1), Scalar(-3), Scalar(0.5)));
    point_vec_.push_back(Point(Scalar(-5), Scalar(-6), Scalar(7)));
  }

  void runAll() {
    bool passed = testLieProperties();
    passed &= testUnity();
    passed &= testRawDataAcces();
    passed &= testConstructors();
    passed &= testSampleUniformSymmetry();
    passed &= testFit();
    processTestResult(passed);
  }

 private:
  bool testLieProperties() {
    LieGroupTests<So3Type> tests(so3_vec_, tangent_vec_, point_vec_);
    return tests.doAllTestsPass();
  }

  bool testUnity() {
    bool passed = true;
    // Test that the complex number magnitude stays close to one.
    So3Type current_q;
    for (size_t i = 0; i < 1000; ++i) {
      for (So3Type const& q : so3_vec_) {
        current_q *= q;
      }
    }
    SOPHUS_TEST_APPROX(
        passed,
        current_q.unitQuaternion().norm(),
        Scalar(1),
        kEpsilon<Scalar>,
        "Magnitude drift");
    return passed;
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 4, 1> raw = {
        Scalar(0), Scalar(1), Scalar(0), Scalar(0)};
    Eigen::Map<So3Type const> map_of_const_so3(raw.data());
    SOPHUS_TEST_APPROX(
        passed,
        map_of_const_so3.unitQuaternion().coeffs().eval(),
        raw,
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        map_of_const_so3.unitQuaternion().coeffs().data(),
        raw.data(),
        "");
    Eigen::Map<So3Type const> const_shallow_copy = map_of_const_so3;
    SOPHUS_TEST_EQUAL(
        passed,
        const_shallow_copy.unitQuaternion().coeffs().eval(),
        map_of_const_so3.unitQuaternion().coeffs().eval(),
        "");

    Eigen::Matrix<Scalar, 4, 1> raw2 = {
        Scalar(1), Scalar(0), Scalar(0), Scalar(0)};
    Eigen::Map<So3Type> map_of_so3(raw.data());
    Eigen::Quaternion<Scalar> quat;
    quat.coeffs() = raw2;
    map_of_so3.setQuaternion(quat);
    SOPHUS_TEST_APPROX(
        passed,
        map_of_so3.unitQuaternion().coeffs().eval(),
        raw2,
        kEpsilon<Scalar>,
        "");
    SOPHUS_TEST_EQUAL(
        passed, map_of_so3.unitQuaternion().coeffs().data(), raw.data(), "");
    SOPHUS_TEST_NEQ(
        passed,
        map_of_so3.unitQuaternion().coeffs().data(),
        quat.coeffs().data(),
        "");
    Eigen::Map<So3Type> shallow_copy = map_of_so3;
    SOPHUS_TEST_EQUAL(
        passed,
        shallow_copy.unitQuaternion().coeffs().eval(),
        map_of_so3.unitQuaternion().coeffs().eval(),
        "");

    So3Type const const_so3(quat);
    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_so3.data()[i], raw2.data()[i], "");
    }

    So3Type so3(quat);
    for (int i = 0; i < 4; ++i) {
      so3.data()[i] = raw[i];
    }

    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, so3.data()[i], raw.data()[i], "");
    }

    SOPHUS_TEST_EQUAL(
        passed,
        So3Type::rotX(Scalar(0.2)).matrix(),
        So3Type::exp(Point(Scalar(0.2), Scalar(0), Scalar(0))).matrix(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        So3Type::rotY(Scalar(-0.2)).matrix(),
        So3Type::exp(Point(Scalar(0), Scalar(-0.2), Scalar(0))).matrix(),
        "");
    SOPHUS_TEST_EQUAL(
        passed,
        So3Type::rotZ(Scalar(1.1)).matrix(),
        So3Type::exp(Point(Scalar(0), Scalar(0), Scalar(1.1))).matrix(),
        "");

    Eigen::Vector4<Scalar> data1(Scalar{1}, Scalar{0}, Scalar{0}, Scalar{0});
    Eigen::Vector4<Scalar> data2(Scalar{0}, Scalar{1}, Scalar{0}, Scalar{0});
    Eigen::Map<So3Type> map1(data1.data());

    Eigen::Map<So3Type> map2(data2.data());

    // map -> map assignment
    map2 = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), map2.matrix(), "");

    // map -> type assignment
    So3Type copy;
    copy = map1;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    // type -> map assignment
    copy = So3Type::rotZ(Scalar(0.5));
    map1 = copy;
    SOPHUS_TEST_EQUAL(passed, map1.matrix(), copy.matrix(), "");

    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Eigen::Matrix3<Scalar> r = so3_vec_.front().matrix();
    So3Type so3(r);
    SOPHUS_TEST_APPROX(passed, r, so3.matrix(), kEpsilon<Scalar>, "");

    return passed;
  }

  template <class TS = Scalar>
  std::enable_if_t<std::is_floating_point<TS>::value, bool>
  testSampleUniformSymmetry() {
    bool passed = true;
    std::default_random_engine generator(0);

    // A non-rigorous test for checking that our sampleUniform() function is
    // giving us symmetric results
    //
    // We (a) split the output space in half, (b) apply a series of random
    // rotations to a point, (c) check which half of the output space each
    // transformed point ends up, and then (d) apply a standard "coin toss"
    // chi-square test

    for (size_t trial = 0; trial < 5; trial++) {
      std::normal_distribution<Scalar> normal(0, 10);

      // Pick a random plane to split the output space by
      Point plane_normal(
          normal(generator), normal(generator), normal(generator));
      plane_normal /= plane_normal.norm();

      // Pick a random point to be rotated
      Point input_point(
          normal(generator), normal(generator), normal(generator));
      input_point /= input_point.norm();

      // Randomly rotate points and track # that land on each side of plane
      size_t positive_count = 0;
      size_t negative_count = 0;
      size_t samples = 5000;
      for (size_t i = 0; i < samples; ++i) {
        So3Type r = So3Type::sampleUniform(generator);
        if (plane_normal.dot(r * input_point) > 0) {
          positive_count++;
        } else {
          negative_count++;
        }
      }

      // Chi-square computation, compare against critical value (p=0.01)
      double expected_count = static_cast<double>(samples) / 2.0;
      double chi_square =
          pow(positive_count - expected_count, 2.0) / expected_count +
          pow(negative_count - expected_count, 2.0) / expected_count;
      SOPHUS_TEST(passed, chi_square < 6.635, "");
    }

    return passed;
  }

  template <class TS = Scalar>
  std::enable_if_t<!std::is_floating_point<TS>::value, bool>
  testSampleUniformSymmetry() {
    return true;
  }

  template <class TS = Scalar>
  std::enable_if_t<std::is_floating_point<TS>::value, bool> testFit() {
    bool passed = true;

    for (int i = 0; i < 100; ++i) {
      Eigen::Matrix3<Scalar> r = Eigen::Matrix3<Scalar>::Random();
      So3Type so3 = So3Type::fitToSo3(r);
      So3Type so3_2 = So3Type::fitToSo3(so3.matrix());

      SOPHUS_TEST_APPROX(
          passed, so3.matrix(), so3_2.matrix(), kEpsilon<Scalar>, "");
    }

    for (Scalar const angle :
         {Scalar(0.0), Scalar(0.1), Scalar(0.3), Scalar(-0.7)}) {
      SOPHUS_TEST_APPROX(
          passed, So3Type::rotX(angle).angleX(), angle, kEpsilon<Scalar>, "");
      SOPHUS_TEST_APPROX(
          passed, So3Type::rotY(angle).angleY(), angle, kEpsilon<Scalar>, "");
      SOPHUS_TEST_APPROX(
          passed, So3Type::rotZ(angle).angleZ(), angle, kEpsilon<Scalar>, "");
    }
    return passed;
  }

  template <class TS = Scalar>
  std::enable_if_t<!std::is_floating_point<TS>::value, bool> testFit() {
    return true;
  }

  std::vector<So3Type, Eigen::aligned_allocator<So3Type>> so3_vec_;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec_;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec_;
};

int testSo3() {
  using std::cerr;
  using std::endl;

  cerr << "Test So3" << endl << endl;
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

int main() { return sophus::testSo3(); }
