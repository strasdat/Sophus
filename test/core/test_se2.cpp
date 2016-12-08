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

#include <sophus/se2.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::SE2Group<double>>;
template class Map<const Sophus::SE2Group<double>>;
}

namespace Sophus {

template class SE2Group<double>;

template <class Scalar>
void tests() {
  using std::vector;
  typedef SO2Group<Scalar> SO2Type;
  typedef SE2Group<Scalar> SE2Type;
  typedef typename SE2Group<Scalar>::Point Point;
  typedef typename SE2Group<Scalar>::Tangent Tangent;
  const Scalar PI = Constants<Scalar>::pi();

  vector<SE2Type, Eigen::aligned_allocator<SE2Type>> se2_vec;
  se2_vec.push_back(SE2Type(SO2Type(0.0), Point(0, 0)));
  se2_vec.push_back(SE2Type(SO2Type(0.2), Point(10, 0)));
  se2_vec.push_back(SE2Type(SO2Type(0.), Point(0, 100)));
  se2_vec.push_back(SE2Type(SO2Type(-1.), Point(20, -1)));
  se2_vec.push_back(
      SE2Type(SO2Type(0.00001), Point(-0.00000001, 0.0000000001)));
  se2_vec.push_back(SE2Type(SO2Type(0.2), Point(0, 0)) *
                    SE2Type(SO2Type(PI), Point(0, 0)) *
                    SE2Type(SO2Type(-0.2), Point(0, 0)));
  se2_vec.push_back(SE2Type(SO2Type(0.3), Point(2, 0)) *
                    SE2Type(SO2Type(PI), Point(0, 0)) *
                    SE2Type(SO2Type(-0.3), Point(0, 6)));

  vector<Tangent, Eigen::aligned_allocator<Tangent>> tangent_vec;
  Tangent tmp;
  tmp << 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 1, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 0, 1, 1;
  tangent_vec.push_back(tmp);
  tmp << -1, 1, 0;
  tangent_vec.push_back(tmp);
  tmp << 20, -1, -1;
  tangent_vec.push_back(tmp);
  tmp << 30, 5, 20;
  tangent_vec.push_back(tmp);

  vector<Point, Eigen::aligned_allocator<Point>> point_vec;
  point_vec.push_back(Point(1, 2));

  GenericTests<SE2Type> tests;
  tests.setGroupElements(se2_vec);
  tests.setTangentVectors(tangent_vec);
  tests.setPoints(point_vec);

  bool passed = tests.doAllTestsPass();
  processTestResult(passed);
}

int test_se2() {
  using std::cerr;
  using std::endl;

  cerr << "Test SE2" << endl << endl;
  cerr << "Double tests: " << endl;
  tests<double>();
  cerr << "Float tests: " << endl;
  tests<float>();
  return 0;
}
}  // namespace Sophus

int main() { return Sophus::test_se2(); }
