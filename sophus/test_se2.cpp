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
#include "se2.hpp"
#include "so3.hpp"

using namespace Sophus;
using namespace std;

template<class Scalar>
bool se2explog_tests() {
  typedef SO2Group<Scalar> SO2Scalar;
  typedef SE2Group<Scalar> SE2Scalar;
  typedef Matrix<Scalar,2,1> Vector2Scalar;
  typedef Matrix<Scalar,3,3> Matrix3Scalar;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();
  const Scalar PI = SophusConstants<Scalar>::pi();

  vector<SE2Scalar> omegas;
  omegas.push_back(SE2Scalar(SO2Scalar(0.0),Vector2Scalar(0,0)));
  omegas.push_back(SE2Scalar(SO2Scalar(0.2),Vector2Scalar(10,0)));
  omegas.push_back(SE2Scalar(SO2Scalar(0.),Vector2Scalar(0,100)));
  omegas.push_back(SE2Scalar(SO2Scalar(-1.),Vector2Scalar(20,-1)));
  omegas.push_back(SE2Scalar(SO2Scalar(0.00001),
                            Vector2Scalar(-0.00000001,0.0000000001)));
  omegas.push_back(SE2Scalar(SO2Scalar(0.2),Vector2Scalar(0,0))
                   *SE2Scalar(SO2Scalar(PI),Vector2Scalar(0,0))
                   *SE2Scalar(SO2Scalar(-0.2),Vector2Scalar(0,0)));
  omegas.push_back(SE2Scalar(SO2Scalar(0.3),Vector2Scalar(2,0))
                   *SE2Scalar(SO2Scalar(PI),Vector2Scalar(0,0))
                   *SE2Scalar(SO2Scalar(-0.3),Vector2Scalar(0,6)));

  bool failed = false;

  for (size_t i=0; i<omegas.size(); ++i) {
    Matrix3Scalar R1 = omegas[i].matrix();
    Matrix3Scalar R2 = SE2Scalar::exp(omegas[i].log()).matrix();
    Matrix3Scalar DiffR = R1-R2;
    Scalar nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "SE2Scalar - exp(log(SE2Scalar))" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<omegas.size(); ++i) {
    Vector2Scalar p(1,2);
    Matrix3Scalar T = omegas[i].matrix();
    Vector2Scalar res1 = omegas[i]*p;
    Vector2Scalar res2
        = T.template topLeftCorner<2,2>()*p
        + T.template topRightCorner<2,1>();
    Scalar nrm = (res1-res2).norm();

    if (isnan(nrm) || nrm>SMALL_EPS)
    {
      cerr << "Transform vector" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (res1-res2) <<endl;
      cerr << endl;
      failed = true;
    }
  }

  for (size_t i=0; i<omegas.size(); ++i) {
    Matrix3Scalar q = omegas[i].matrix();
    Matrix3Scalar inv_q = omegas[i].inverse().matrix();
    Matrix3Scalar res = q*inv_q ;
    Matrix3Scalar I;
    I.setIdentity();

    Scalar nrm = (res-I).norm();

    if (isnan(nrm) || nrm>SMALL_EPS)
    {
      cerr << "Inverse" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (res-I) <<endl;
      cerr << endl;
      failed = true;
    }
  }

  for (size_t i=0; i<omegas.size(); ++i) {
    for (size_t j=0; j<omegas.size(); ++j) {
      Matrix3Scalar mul_resmat = (omegas[i]*omegas[j]).matrix();
      Scalar fastmul_res_raw[SE2Scalar::num_parameters];
      Eigen::Map<SE2Scalar> fastmul_res(fastmul_res_raw);
      fastmul_res = omegas[i];
      fastmul_res.fastMultiply(omegas[j]);
      Matrix3Scalar diff =  mul_resmat-fastmul_res.matrix();
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

template<class Scalar>
bool se2bracket_tests() {
  typedef SE2Group<Scalar> SE2Scalar;
  typedef Matrix<Scalar,3,1> Vector3Scalar;
  typedef Matrix<Scalar,3,3> Matrix3Scalar;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();

  bool failed = false;
  vector<Vector3Scalar> vecs;
  Vector3Scalar tmp;
  tmp << 0,0,0;
  vecs.push_back(tmp);
  tmp << 1,0,0;
  vecs.push_back(tmp);
  tmp << 0,1,1;
  vecs.push_back(tmp);
  tmp << -1,1,0;
  vecs.push_back(tmp);
  tmp << 20,-1,-1;
  vecs.push_back(tmp);
  tmp << 30,5,20;
  vecs.push_back(tmp);
  for (size_t i=0; i<vecs.size(); ++i) {
    Vector3Scalar resDiff = vecs[i] - SE2Scalar::vee(SE2Scalar::hat(vecs[i]));
    if (resDiff.norm()>SMALL_EPS)
    {
      cerr << "Hat-vee Test" << endl;
      cerr  << "Test case: " << i <<  endl;
      cerr << resDiff.transpose() << endl;
      cerr << endl;
    }

    for (size_t j=0; j<vecs.size(); ++j) {
      Vector3Scalar res1 = SE2Scalar::lieBracket(vecs[i],vecs[j]);
      Matrix3Scalar hati = SE2Scalar::hat(vecs[i]);
      Matrix3Scalar hatj = SE2Scalar::hat(vecs[j]);

      Vector3Scalar res2 = SE2Scalar::vee(hati*hatj-hatj*hati);
      Vector3Scalar resDiff = res1-res2;
      if (resDiff.norm()>SMALL_EPS) {
        cerr << "SE2Scalar Lie Bracket Test" << endl;
        cerr  << "Test case: " << i << ", " <<j<< endl;
        cerr << vecs[i].transpose() << endl;
        cerr << vecs[j].transpose() << endl;
        cerr << res1 << endl;
        cerr << res2 << endl;
        cerr << resDiff.transpose() << endl;
        cerr << endl;
        failed = true;
      }
    }

    Vector3Scalar omega = vecs[i];
    Matrix3Scalar exp_x = SE2Scalar::exp(omega).matrix();
    Matrix3Scalar expmap_hat_x = (SE2Scalar::hat(omega)).exp();
    Matrix3Scalar DiffR = exp_x-expmap_hat_x;
    Scalar nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "expmap(hat(x)) - exp(x)" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << exp_x <<endl;
      cerr << expmap_hat_x <<endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }

  return failed;
}



int main() {
  cerr << "Test SE2" << endl << endl;

  cerr << "Double tests: " << endl;
  bool failed = se2explog_tests<double>();
  failed = failed || se2bracket_tests<double>();
  if (failed) {
    cerr << "failed!" << endl << endl;
    exit(-1);
  } else {
    cerr << "passed." << endl << endl;
  }

  cerr << "Float tests: " << endl;
  failed = failed || se2explog_tests<float>();
  failed = failed || se2bracket_tests<float>();
  if (failed) {
    cerr << "failed!" << endl << endl;
    exit(-1);
  } else {
    cerr << "passed." << endl << endl;
  }
  return 0;
}
