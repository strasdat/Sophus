// This file is part of Sophus.
//
// Copyright 2013-2014 Ping-Lin Chang
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
#include <vector>

#include "sophus/rxso2.hpp"
#include "tests.hpp"

using namespace Sophus;
using namespace std;

template<class Scalar>
void tests() {

  typedef RxSO2Group<Scalar> RxSO2Type;
  typedef typename RxSO2Group<Scalar>::Point Point;
  typedef typename RxSO2Group<Scalar>::Tangent Tangent;

  vector<RxSO2Type> rxso2_vec;
  rxso2_vec.push_back(RxSO2Type::exp(Tangent(0.0, 0.0)));
  rxso2_vec.push_back(RxSO2Type::exp(Tangent(0.2, 1.0)));
  rxso2_vec.push_back(RxSO2Type::exp(Tangent(10., -1.1)));
  rxso2_vec.push_back(RxSO2Type::exp(Tangent(0.00001, -5.0)));
  rxso2_vec.push_back(RxSO2Type::exp(Tangent(M_PI, 1.54)));
  rxso2_vec.push_back(RxSO2Type::exp(Tangent(0.2, 0.0))
                      *RxSO2Type::exp(Tangent(M_PI, 1.0))
                      *RxSO2Type::exp(Tangent(-0.2, 2.0)));
  rxso2_vec.push_back(RxSO2Type::exp(Tangent(-0.3, 0.0))
                      *RxSO2Type::exp(Tangent(M_PI, 3.0))
                      *RxSO2Type::exp(Tangent(0.3, 0.0)));

  vector<Tangent> tangent_vec;
  Tangent tmp;
  tmp << 0,0;
  tangent_vec.push_back(tmp);
  tmp << 1,0;
  tangent_vec.push_back(tmp);
  tmp << 1,0.1;
  tangent_vec.push_back(tmp);
  tmp << 0,0.1;
  tangent_vec.push_back(tmp);
  tmp << 0,-0.1;
  tangent_vec.push_back(tmp);
  tmp << -1,-0.8;
  tangent_vec.push_back(tmp);
  tmp << 20,2;
  tangent_vec.push_back(tmp);

  vector<Point> point_vec;
  point_vec.push_back(Point(1,2));

  Tests<RxSO2Type> tests;
  tests.setGroupElements(rxso2_vec);
  tests.setTangentVectors(tangent_vec);
  tests.setPoints(point_vec);

  tests.runAllTests();
}

int main() {
  cerr << "Test RxSO2" << endl << endl;

  cerr << "Double tests: " << endl;
  tests<double>();

  cerr << "Float tests: " << endl;
  tests<float>();
  return 0;
}
