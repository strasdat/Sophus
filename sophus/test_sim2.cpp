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

#include <unsupported/Eigen/MatrixFunctions>

#include "sim2.hpp"
#include "tests.hpp"

using namespace Sophus;
using namespace std;

template<class Scalar>
void tests() {

  typedef Sim2Group<Scalar> Sim2Type;
  typedef RxSO2Group<Scalar> RxSO2Type;
  typedef typename Sim2Group<Scalar>::Point Point;
  typedef typename Sim2Group<Scalar>::Tangent Tangent;
  typedef Matrix<Scalar,2,1> Vector2Type;

  vector<Sim2Type> sim2_vec;
  sim2_vec.push_back(Sim2Type(RxSO2Type::exp(Vector2Type(0.2, 1.)),
                              Point(0,0)));
  sim2_vec.push_back(Sim2Type(RxSO2Type::exp(Vector2Type(0.5, 1.1)),
                              Point(10,0)));
  sim2_vec.push_back(Sim2Type(RxSO2Type::exp(Vector2Type(0.,1.1)),
                              Point(0,10)));
  sim2_vec.push_back(Sim2Type(RxSO2Type::exp(Vector2Type(0.00001, 0.0)),
                              Point(0,0)));
  sim2_vec.push_back(Sim2Type(RxSO2Type::exp(
                                Vector2Type(0.00001, 0.00000001)),
                              Point(1,-1.00000001)));
  sim2_vec.push_back(Sim2Type(RxSO2Type::exp(Vector2Type(0.00001, 0)),
                              Point(0.01,0)));
  sim2_vec.push_back(Sim2Type(RxSO2Type::exp(Vector2Type(M_PI, 0.9)),
                              Point(4,-5)));
  sim2_vec.push_back(Sim2Type(RxSO2Type::exp(Vector2Type(0.2, 0.5)),
                              Point(0,0))
                     *Sim2Type(RxSO2Type::exp(Vector2Type(M_PI, 0)),
                               Point(0,0))
                     *Sim2Type(RxSO2Type::exp(Vector2Type(-0.2, -0.5)),
                               Point(0,0)));
  sim2_vec.push_back(Sim2Type(RxSO2Type::exp(Vector2Type(0.3, -1)),
                              Point(2,-7))
                     *Sim2Type(RxSO2Type::exp(Vector2Type(M_PI, 0)),
                               Point(0,0))
                     *Sim2Type(RxSO2Type::exp(Vector2Type(-0.3, 0)),
                               Point(0,6)));
  vector<Tangent> tangent_vec;
  Tangent tmp;
  tmp << 0,0,0,0;
  tangent_vec.push_back(tmp);
  tmp << 1,0,0,0;
  tangent_vec.push_back(tmp);
  tmp << 0,1,0,0.1;
  tangent_vec.push_back(tmp);
  tmp << 0,1,1,0.1;
  tangent_vec.push_back(tmp);
  tmp << -1,1,1,-0.1;
  tangent_vec.push_back(tmp);
  tmp << 20,-1,-1,-0.1;
  tangent_vec.push_back(tmp);
  tmp << 30,-1,20,1.5;
  tangent_vec.push_back(tmp);


  vector<Point> point_vec;
  point_vec.push_back(Point(1,4));

  Tests<Sim2Type> tests;
  tests.setGroupElements(sim2_vec);
  tests.setTangentVectors(tangent_vec);
  tests.setPoints(point_vec);

  tests.runAllTests();
}

int main() {
  cerr << "Test Sim2" << endl << endl;

  cerr << "Double tests: " << endl;
  tests<double>();

  cerr << "Float tests: " << endl;
  tests<float>();
  return 0;
}
