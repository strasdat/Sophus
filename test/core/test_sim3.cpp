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

#include <unsupported/Eigen/MatrixFunctions>

#include <sophus/sim3.hpp>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::Sim3Group<double>>;
template class Map<const Sophus::Sim3Group<double>>;
}

namespace Sophus {

template class Sim3Group<double>;

template <class Scalar>
void tests() {
  using std::vector;
  typedef Sim3Group<Scalar> Sim3Type;
  typedef RxSO3Group<Scalar> RxSO3Type;
  typedef typename Sim3Group<Scalar>::Point Point;
  typedef typename Sim3Group<Scalar>::Tangent Tangent;
  typedef Eigen::Matrix<Scalar, 4, 1> Vector4Type;
  const Scalar PI = Constants<Scalar>::pi();

  vector<Sim3Type, Eigen::aligned_allocator<Sim3Type>> sim3_vec;
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0.2, 0.5, 0.0, 1.)), Point(0, 0, 0)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0.2, 0.5, -1.0, 1.1)),
                              Point(10, 0, 0)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0., 0.)), Point(0, 10, 5)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0., 1.1)), Point(0, 10, 5)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0.00001, 0.)),
                              Point(0, 0, 0)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0.00001, 0.0000001)),
               Point(1, -1.00000001, 2.0000000001)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0.00001, 0)),
                              Point(0.01, 0, 0)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(PI, 0, 0, 0.9)), Point(4, -5, 0)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0.2, 0.5, 0.0, 0)), Point(0, 0, 0)) *
      Sim3Type(RxSO3Type::exp(Vector4Type(PI, 0, 0, 0)), Point(0, 0, 0)) *
      Sim3Type(RxSO3Type::exp(Vector4Type(-0.2, -0.5, -0.0, 0)),
               Point(0, 0, 0)));
  sim3_vec.push_back(
      Sim3Type(RxSO3Type::exp(Vector4Type(0.3, 0.5, 0.1, 0)), Point(2, 0, -7)) *
      Sim3Type(RxSO3Type::exp(Vector4Type(PI, 0, 0, 0)), Point(0, 0, 0)) *
      Sim3Type(RxSO3Type::exp(Vector4Type(-0.3, -0.5, -0.1, 0)),
               Point(0, 6, 0)));
  vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec;
  Tangent tmp;
  tmp << 0, 0, 0, 0, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 1, 0, 0, 0, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 0, 1, 0, 1, 0, 0, 0.1;
  tangent_vec.push_back(tmp);
  tmp << 0, 0, 1, 0, 1, 0, 0.1;
  tangent_vec.push_back(tmp);
  tmp << -1, 1, 0, 0, 0, 1, -0.1;
  tangent_vec.push_back(tmp);
  tmp << 20, -1, 0, -1, 1, 0, -0.1;
  tangent_vec.push_back(tmp);
  tmp << 30, 5, -1, 20, -1, 0, 1.5;
  tangent_vec.push_back(tmp);

  vector<Point, Eigen::aligned_allocator<Point>> point_vec;
  point_vec.push_back(Point(1, 2, 4));

  GenericTests<Sim3Type> tests;
  tests.setGroupElements(sim3_vec);
  tests.setTangentVectors(tangent_vec);
  tests.setPoints(point_vec);

  bool passed = tests.doAllTestsPass();

  Sim3Type sim3;
  Scalar scale(1.2);
  sim3.setScale(scale);
  SOPHUS_TEST_APPROX(passed, scale, sim3.scale(), Constants<Scalar>::epsilon(),
                     "setScale");
  processTestResult(passed);
}

int test_sim3() {
  using std::cerr;
  using std::endl;

  cerr << "Test Sim3" << endl << endl;
  cerr << "Double tests: " << endl;
  tests<double>();
  cerr << "Float tests: " << endl;
  tests<float>();
  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_sim3(); }
