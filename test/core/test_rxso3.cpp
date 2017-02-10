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

#include <sophus/rxso3.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::RxSO3<double>>;
template class Map<const Sophus::RxSO3<double>>;
}

namespace Sophus {

template class RxSO3<double>;

template <class Scalar>
void tests() {
  using std::vector;
  typedef RxSO3<Scalar> RxSO3Type;
  typedef typename RxSO3<Scalar>::Point Point;
  typedef typename RxSO3<Scalar>::Tangent Tangent;

  const Scalar PI = Constants<Scalar>::pi();

  vector<RxSO3Type, Eigen::aligned_allocator<RxSO3Type>> rxso3_vec;
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0.2, 0.5, 0.0, 1.)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0.2, 0.5, -1.0, 1.1)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0., 0., 0., 1.1)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0., 0., 0.00001, 0.)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0., 0., 0.00001, 0.00001)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0., 0., 0.00001, 0)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(PI, 0, 0, 0.9)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0.2, 0.5, 0.0, 0)) *
                      RxSO3Type::exp(Tangent(PI, 0, 0, 0.0)) *
                      RxSO3Type::exp(Tangent(-0.2, -0.5, -0.0, 0)));
  rxso3_vec.push_back(RxSO3Type::exp(Tangent(0.3, 0.5, 0.1, 0)) *
                      RxSO3Type::exp(Tangent(PI, 0, 0, 0)) *
                      RxSO3Type::exp(Tangent(-0.3, -0.5, -0.1, 0)));

  vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec;
  Tangent tmp;
  tmp << 0, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 1, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 1, 0, 0, 0.1;
  tangent_vec.push_back(tmp);
  tmp << 0, 1, 0, 0.1;
  tangent_vec.push_back(tmp);
  tmp << 0, 0, 1, -0.1;
  tangent_vec.push_back(tmp);
  tmp << -1, 1, 0, -0.1;
  tangent_vec.push_back(tmp);
  tmp << 20, -1, 0, 2;
  tangent_vec.push_back(tmp);

  vector<Point, Eigen::aligned_allocator<Point>> point_vec;
  point_vec.push_back(Point(1, 2, 4));

  GenericTests<RxSO3Type> tests;
  tests.setGroupElements(rxso3_vec);
  tests.setTangentVectors(tangent_vec);
  tests.setPoints(point_vec);

  bool passed = tests.doAllTestsPass();

  // TODO: Add proper unit tests for all functions.
  RxSO3Type rxso3;
  Scalar scale(1.2);
  rxso3.setScale(scale);
  SOPHUS_TEST_APPROX(passed, scale, rxso3.scale(), Constants<Scalar>::epsilon(),
                     "setScale");
  auto so3 = rxso3_vec[0].so3();
  rxso3.setSO3(so3);
  SOPHUS_TEST_APPROX(passed, scale, rxso3.scale(), Constants<Scalar>::epsilon(),
                     "setScale");
  SOPHUS_TEST_APPROX(passed, RxSO3Type(scale, so3).matrix(), rxso3.matrix(),
                     Constants<Scalar>::epsilon(), "RxSO3(scale, SO3)");
  SOPHUS_TEST_APPROX(passed, RxSO3Type(scale, so3.matrix()).matrix(),
                     rxso3.matrix(), Constants<Scalar>::epsilon(),
                     "RxSO3(scale, SO3)");
  Eigen::Matrix<Scalar, 3, 3> R =
      SO3<Scalar>::exp(Point(0.2, 0.5, -1.0)).matrix();
  Eigen::Matrix<Scalar, 3, 3> sR = R * Scalar(1.3);
  rxso3.setScaledRotationMatrix(sR);
  SOPHUS_TEST_APPROX(passed, sR, rxso3.matrix(), Constants<Scalar>::epsilon(),
                     "setScaleRotationMatrix");
  rxso3.setScale(scale);
  rxso3.setRotationMatrix(R);
  SOPHUS_TEST_APPROX(passed, R, rxso3.rotationMatrix(),
                     Constants<Scalar>::epsilon(), "setRotationMatrix");
  SOPHUS_TEST_APPROX(passed, scale, rxso3.scale(), Constants<Scalar>::epsilon(),
                     "setScale");
  processTestResult(passed);
}

int test_rxso3() {
  using std::cerr;
  using std::endl;

  cerr << "Test RxSO3" << endl << endl;
  cerr << "Double tests: " << endl;
  tests<double>();
  cerr << "Float tests: " << endl;
  tests<float>();
  return 0;
}

}  // Sophus

int main() { return Sophus::test_rxso3(); }
