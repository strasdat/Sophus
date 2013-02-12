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
  typedef SO2Group<Scalar> SO2Type;
  typedef Matrix<Scalar,2,1> Vector2Type;
  typedef typename SO2Group<Scalar>::TransformationType TransformationType;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();
  const Scalar PI = SophusConstants<Scalar>::pi();

  vector<SO2Type> so2;
  so2.push_back(SO2Type::exp(0.0));
  so2.push_back(SO2Type::exp(0.2));
  so2.push_back(SO2Type::exp(10.));
  so2.push_back(SO2Type::exp(0.00001));
  so2.push_back(SO2Type::exp(PI));
  so2.push_back(SO2Type::exp(0.2)
                *SO2Type::exp(PI)
                *SO2Type::exp(-0.2));
  so2.push_back(SO2Type::exp(-0.3)
                *SO2Type::exp(PI)
                *SO2Type::exp(0.3));

  bool failed = false;

  for (size_t i=0; i<so2.size(); ++i) {
    TransformationType R1 = so2[i].matrix();
    TransformationType R2 = SO2Type::exp(so2[i].log()).matrix();

    TransformationType DiffR = R1-R2;
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
    Vector2Type p(1,2);
    TransformationType R = so2[i].matrix();
    Vector2Type res1 = so2[i]*p;
    Vector2Type res2 = R*p;

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
    TransformationType q = so2[i].matrix();
    TransformationType inv_q = so2[i].inverse().matrix();
    TransformationType res = q*inv_q ;
    TransformationType I;
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
    TransformationType exp_x = SO2Type::exp(omega).matrix();
    TransformationType expmap_hat_x = (SO2Type::hat(omega)).exp();
    TransformationType DiffR = exp_x-expmap_hat_x;
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
      TransformationType mul_resmat = (so2[i]*so2[j]).matrix();
      Scalar fastmul_res_raw[SO2Type::num_parameters];
      Eigen::Map<SO2Type> fastmul_res(fastmul_res_raw);
      fastmul_res = so2[i];
      fastmul_res.fastMultiply(so2[j]);
      TransformationType diff =  mul_resmat-fastmul_res.matrix();
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
