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
#include <vector>

#include <unsupported/Eigen/MatrixFunctions>

#include "so2.hpp"
#include "so3.hpp"

using namespace Sophus;
using namespace std;

template<class Scalar>
bool so2explog_tests() {
  typedef SO2Group<Scalar> SO2Scalar;
  typedef Quaternion<Scalar> QuaternionScalar;
  typedef Matrix<Scalar,2,1> Vector2Scalar;
  typedef Matrix<Scalar,2,2> Matrix2Scalar;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();
  const Scalar PI = SophusConstants<Scalar>::pi();

  vector<SO2Scalar> so2;
  so2.push_back(SO2Scalar::exp(0.0));
  so2.push_back(SO2Scalar::exp(0.2));
  so2.push_back(SO2Scalar::exp(10.));
  so2.push_back(SO2Scalar::exp(0.00001));
  so2.push_back(SO2Scalar::exp(PI));
  so2.push_back(SO2Scalar::exp(0.2)
                *SO2Scalar::exp(PI)
                *SO2Scalar::exp(-0.2));
  so2.push_back(SO2Scalar::exp(-0.3)
                *SO2Scalar::exp(PI)
                *SO2Scalar::exp(0.3));

  bool failed = false;

  for (size_t i=0; i<so2.size(); ++i) {
    Matrix2Scalar R1 = so2[i].matrix();
    Matrix2Scalar R2 = SO2Scalar::exp(so2[i].log()).matrix();

    Matrix2Scalar DiffR = R1-R2;
    double nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "SO2 - exp(log(SO2))" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }

  for (size_t i=0; i<so2.size(); ++i) {
    Vector2Scalar p(1,2);
    Matrix2Scalar R = so2[i].matrix();
    Vector2Scalar res1 = so2[i]*p;
    Vector2Scalar res2 = R*p;

    Scalar nrm = (res1-res2).norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "Transform vector" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (res1-res2) <<endl;
      cerr << endl;
      failed = true;
    }
  }

  for (size_t i=0; i<so2.size(); ++i) {
    Matrix2Scalar q = so2[i].matrix();
    Matrix2Scalar inv_q = so2[i].inverse().matrix();
    Matrix2Scalar res = q*inv_q ;
    Matrix2Scalar I;
    I.setIdentity();

    Scalar nrm = (res-I).norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "Inverse" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (res-I) <<endl;
      cerr << endl;
      failed = true;
    }
  }

  for (size_t i=0; i<so2.size(); ++i) {
    Scalar omega = so2[i].log();
    Matrix2Scalar exp_x = SO2Scalar::exp(omega).matrix();
    Matrix2Scalar expmap_hat_x = (SO2Scalar::hat(omega)).exp();
    Matrix2Scalar DiffR = exp_x-expmap_hat_x;
    Scalar nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "expmap(hat(x)) - exp(x)" << endl;
      cerr  << "Test case: " << i << endl;
      //      cerr << exp_x <<endl;
      //      cerr << expmap_hat_x <<endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }

  for (size_t i=0; i<so2.size(); ++i) {
    for (size_t j=0; j<so2.size(); ++j) {
      Matrix2Scalar mul_resmat = (so2[i]*so2[j]).matrix();
      Scalar fastmul_res_raw[SO2Scalar::num_parameters];
      Eigen::Map<SO2Scalar> fastmul_res(fastmul_res_raw);
      fastmul_res = so2[i];
      fastmul_res.fastMultiply(so2[j]);
      Matrix2Scalar diff =  mul_resmat-fastmul_res.matrix();
      Scalar nrm = diff.norm();
      if (isnan(nrm) || nrm>SMALL_EPS) {
        cerr << "Fast multiplication" << endl;
        cerr  << "Test case: " << i  << "," << j << endl;
        cerr << diff <<endl;
        cerr << endl;
        failed = true;
      }
    }
  }
  return failed;
}

int main() {
  cerr << "Test SO2" << endl << endl;
  cerr << "Double tests: " << endl;
  bool failed = so2explog_tests<double>();
  if (failed) {
    cerr << "failed!" << endl << endl;
    exit(-1);
  } else {
    cerr << "passed." << endl << endl;
  }

  cerr << "Float tests: " << endl;
  failed = failed || so2explog_tests<float>();
  if (failed) {
    cerr << "failed!" << endl << endl;
    exit(-1);
  } else {
    cerr << "passed." << endl << endl;
  }
  return 0;
}
