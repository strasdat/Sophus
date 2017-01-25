// This file is part of Sophus.
//
// Copyright 2012-2013 Hauke Strasdat
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <iostream>

#include <sophus/so3.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::SO3<double>>;
template class Map<const Sophus::SO3<double>>;
}

namespace Sophus {

template class SO3<double>;

template <class Scalar>
class Tests {
 public:
  using SO3Type = SO3<Scalar>;
  using Point = typename SO3<Scalar>::Point;
  using Tangent = typename SO3<Scalar>::Tangent;
  const Scalar PI = Constants<Scalar>::pi();

  Tests() {
    so3_vec.push_back(SO3Type(Eigen::Quaternion<Scalar>(0.1e-11, 0., 1., 0.)));
    so3_vec.push_back(
        SO3Type(Eigen::Quaternion<Scalar>(-1, 0.00001, 0.0, 0.0)));
    so3_vec.push_back(SO3Type::exp(Point(0.2, 0.5, 0.0)));
    so3_vec.push_back(SO3Type::exp(Point(0.2, 0.5, -1.0)));
    so3_vec.push_back(SO3Type::exp(Point(0., 0., 0.)));
    so3_vec.push_back(SO3Type::exp(Point(0., 0., 0.00001)));
    so3_vec.push_back(SO3Type::exp(Point(PI, 0, 0)));
    so3_vec.push_back(SO3Type::exp(Point(0.2, 0.5, 0.0)) *
                      SO3Type::exp(Point(PI, 0, 0)) *
                      SO3Type::exp(Point(-0.2, -0.5, -0.0)));
    so3_vec.push_back(SO3Type::exp(Point(0.3, 0.5, 0.1)) *
                      SO3Type::exp(Point(PI, 0, 0)) *
                      SO3Type::exp(Point(-0.3, -0.5, -0.1)));
    tangent_vec.push_back(Tangent(0, 0, 0));
    tangent_vec.push_back(Tangent(1, 0, 0));
    tangent_vec.push_back(Tangent(0, 1, 0));
    tangent_vec.push_back(Tangent(PI / 2., PI / 2., 0.0));
    tangent_vec.push_back(Tangent(-1, 1, 0));
    tangent_vec.push_back(Tangent(20, -1, 0));
    tangent_vec.push_back(Tangent(30, 5, -1));

    point_vec.push_back(Point(1, 2, 4));
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
    GenericTests<SO3Type> tests;
    tests.setGroupElements(so3_vec);
    tests.setTangentVectors(tangent_vec);
    tests.setPoints(point_vec);
    return tests.doAllTestsPass();
  }

  bool testUnity() {
    bool passed = true;
    // Test that the complex number magnitude stays close to one.
    SO3Type current_q;
    for (std::size_t i = 0; i < 1000; ++i) {
      for (const auto& q : so3_vec) {
        current_q *= q;
      }
    }
    SOPHUS_TEST_APPROX(passed, current_q.unit_quaternion().norm(), Scalar(1),
                       Constants<Scalar>::epsilon(), "Magnitude drift");
    return passed;
  }

  bool testRawDataAcces() {
    bool passed = true;
    Eigen::Matrix<Scalar, 4, 1> raw = {0, 1, 0, 0};
    Eigen::Map<const SO3Type> const_so3_map(raw.data());
    SOPHUS_TEST_APPROX(passed, const_so3_map.unit_quaternion().coeffs().eval(),
                       raw, Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, const_so3_map.unit_quaternion().coeffs().data(),
                      raw.data());
    Eigen::Map<const SO3Type> const_shallow_copy = const_so3_map;
    SOPHUS_TEST_EQUAL(passed,
                      const_shallow_copy.unit_quaternion().coeffs().eval(),
                      const_so3_map.unit_quaternion().coeffs().eval());

    Eigen::Matrix<Scalar, 4, 1> raw2 = {1, 0, 0, 0};
    Eigen::Map<SO3Type> so3_map(raw.data());
    Eigen::Quaternion<Scalar> quat;
    quat.coeffs() = raw2;
    so3_map.setQuaternion(quat);
    SOPHUS_TEST_APPROX(passed, so3_map.unit_quaternion().coeffs().eval(), raw2,
                       Constants<Scalar>::epsilon());
    SOPHUS_TEST_EQUAL(passed, so3_map.unit_quaternion().coeffs().data(),
                      raw.data());
    SOPHUS_TEST_NEQ(passed, so3_map.unit_quaternion().coeffs().data(),
                    quat.coeffs().data());
    Eigen::Map<SO3Type> shallow_copy = so3_map;
    SOPHUS_TEST_EQUAL(passed, shallow_copy.unit_quaternion().coeffs().eval(),
                      so3_map.unit_quaternion().coeffs().eval());

    const SO3Type const_so3(quat);
    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, const_so3.data()[i], raw2.data()[i]);
    }

    SO3Type so3(quat);
    for (int i = 0; i < 4; ++i) {
      so3.data()[i] = raw[i];
    }

    for (int i = 0; i < 4; ++i) {
      SOPHUS_TEST_EQUAL(passed, so3.data()[i], raw.data()[i]);
    }
    return passed;
  }

  bool testConstructors() {
    bool passed = true;
    Eigen::Matrix<Scalar, 3, 3> R = so3_vec.front().matrix();
    SO3Type so3(R);
    SOPHUS_TEST_APPROX(passed, R, so3.matrix(), Constants<Scalar>::epsilon());
    return passed;
  }

  std::vector<SO3Type, Eigen::aligned_allocator<SO3Type>> so3_vec;
  std::vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec;
  std::vector<Point, Eigen::aligned_allocator<Point>> point_vec;
};

int test_so3() {
  using std::cerr;
  using std::endl;

  cerr << "Test SO3" << endl << endl;
  cerr << "Double tests: " << endl;
  Tests<double>().runAll();
  cerr << "Float tests: " << endl;
  Tests<float>().runAll();
  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_so3(); }
