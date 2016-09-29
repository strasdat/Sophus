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

// These definitions are not standard C++ and are missing on some compilers.
#if !defined(M_PI) || !defined(M_PI_2)
#define M_PI      3.14159265358979323846264338328
#define M_PI_2    1.57079632679489661923132169164
#endif

#include <sophus/so2.hpp>
#include "tests.hpp"

using namespace Sophus;
using namespace std;

template<class Scalar>
void tests() {

  typedef SO2Group<Scalar> SO2Type;
  typedef typename SO2Group<Scalar>::Point Point;
  typedef typename SO2Group<Scalar>::Tangent Tangent;

  vector<SO2Type, Eigen::aligned_allocator<SO2Type> > so2_vec;
  so2_vec.push_back(SO2Type::exp(0.0));
  so2_vec.push_back(SO2Type::exp(0.2));
  so2_vec.push_back(SO2Type::exp(10.));
  so2_vec.push_back(SO2Type::exp(0.00001));
  so2_vec.push_back(SO2Type::exp(M_PI));
  so2_vec.push_back(SO2Type::exp(0.2)
                    *SO2Type::exp(M_PI)
                    *SO2Type::exp(-0.2));
  so2_vec.push_back(SO2Type::exp(-0.3)
                    *SO2Type::exp(M_PI)
                    *SO2Type::exp(0.3));

  vector<Tangent, Eigen::aligned_allocator<Tangent> > tangent_vec;
  tangent_vec.push_back(Tangent(0));
  tangent_vec.push_back(Tangent(1));
  tangent_vec.push_back(Tangent(M_PI_2));
  tangent_vec.push_back(Tangent(-1));
  tangent_vec.push_back(Tangent(20));
  tangent_vec.push_back(Tangent(M_PI_2+0.0001));

  vector<Point, Eigen::aligned_allocator<Point> > point_vec;
  point_vec.push_back(Point(1,2));

  Tests<SO2Type> tests;
  tests.setGroupElements(so2_vec);
  tests.setTangentVectors(tangent_vec);
  tests.setPoints(point_vec);

  tests.runAllTests();
}

int test_so2() {
  cerr << "Test SO2" << endl << endl;

  cerr << "Double tests: " << endl;
  tests<double>();

  cerr << "Float tests: " << endl;
  tests<float>();
  return 0;
}

int main() {
  return test_so2();
}
