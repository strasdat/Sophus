// This file is part of Sophus.
//
// Copyright 2012 Hauke Strasdat
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

#include "so3.h"

using namespace Sophus;
using namespace std;

template<class Scalar>
bool so3explog_tests() {
  typedef SO3Group<Scalar> SO3Scalar;
  typedef Quaternion<Scalar> QuaternionScalar;
  typedef Matrix<Scalar,3,1> Vector3Scalar;
  typedef Matrix<Scalar,3,3> Matrix3Scalar;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();
  const Scalar PI = SophusConstants<Scalar>::pi();

  vector<SO3Scalar> omegas;
  omegas.push_back(SO3Scalar(QuaternionScalar(0.1e-11, 0., 1., 0.)));
  omegas.push_back(SO3Scalar(QuaternionScalar(-1,0.00001,0.0,0.0)));
  omegas.push_back(SO3Scalar::exp(Vector3Scalar(0.2, 0.5, 0.0)));
  omegas.push_back(SO3Scalar::exp(Vector3Scalar(0.2, 0.5, -1.0)));
  omegas.push_back(SO3Scalar::exp(Vector3Scalar(0., 0., 0.)));
  omegas.push_back(SO3Scalar::exp(Vector3Scalar(0., 0., 0.00001)));
  omegas.push_back(SO3Scalar::exp(Vector3Scalar(M_PI, 0, 0)));
  omegas.push_back(SO3Scalar::exp(Vector3Scalar(0.2, 0.5, 0.0))
                   *SO3Scalar::exp(Vector3Scalar(M_PI, 0, 0))
                   *SO3Scalar::exp(Vector3Scalar(-0.2, -0.5, -0.0)));
  omegas.push_back(SO3Scalar::exp(Vector3Scalar(0.3, 0.5, 0.1))
                   *SO3Scalar::exp(Vector3Scalar(M_PI, 0, 0))
                   *SO3Scalar::exp(Vector3Scalar(-0.3, -0.5, -0.1)));

  bool failed = false;

  for (size_t i=0; i<omegas.size(); ++i) {
    Matrix3Scalar R1 = omegas[i].matrix();
    Scalar theta;
    Matrix3Scalar R2
        = SO3Scalar::exp(SO3Scalar::logAndTheta(omegas[i],&theta)).matrix();

    Matrix3Scalar DiffR = R1-R2;
    Scalar nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "SO3 - exp(log(SO3))" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }

    if (theta>PI || theta<-PI) {
      cerr << "log theta not in [-pi,pi]" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << theta <<endl;
      cerr << endl;
      failed = true;
    }
  }

  for (size_t i=0; i<omegas.size(); ++i) {
    Vector3Scalar p(1,2,4);
    Matrix3Scalar sR = omegas[i].matrix();
    Vector3Scalar res1 = omegas[i]*p;
    Vector3Scalar res2 = sR*p;

    Scalar nrm = (res1-res2).norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
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

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "Inverse" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (res-I) <<endl;
      cerr << endl;
      failed = true;
    }
  }
  return failed;
}

template<class Scalar>
bool so3bracket_tests() {
  typedef SO3Group<Scalar> SO3Scalar;
  typedef Matrix<Scalar,3,1> Vector3Scalar;
  typedef Matrix<Scalar,3,3> Matrix3Scalar;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();

  bool failed = false;
  vector<Vector3Scalar> vecs;
  vecs.push_back(Vector3Scalar(0,0,0));
  vecs.push_back(Vector3Scalar(1,0,0));
  vecs.push_back(Vector3Scalar(0,1,0));
  vecs.push_back(Vector3Scalar(M_PI_2,M_PI_2,0.0));
  vecs.push_back(Vector3Scalar(-1,1,0));
  vecs.push_back(Vector3Scalar(20,-1,0));
  vecs.push_back(Vector3Scalar(30,5,-1));
  for (unsigned int i=0; i<vecs.size(); ++i) {
    for (unsigned int j=0; j<vecs.size(); ++j) {
      Vector3Scalar res1 = SO3Scalar::lieBracket(vecs[i],vecs[j]);
      Matrix3Scalar mat =
          SO3Scalar::hat(vecs[i])*SO3Scalar::hat(vecs[j])
          -SO3Scalar::hat(vecs[j])*SO3Scalar::hat(vecs[i]);
      Vector3Scalar res2 = SO3Scalar::vee(mat);
      Vector3Scalar resDiff = res1-res2;
      if (resDiff.norm()>SMALL_EPS) {
        cerr << "SO3 Lie Bracket Test" << endl;
        cerr  << "Test case: " << i << ", " <<j<< endl;
        cerr << res1-res2 << endl;
        cerr << endl;
        failed = true;
      }
    }

    Vector3Scalar omega = vecs[i];
    Matrix3Scalar exp_x = SO3Scalar::exp(omega).matrix();
    Matrix3Scalar expmap_hat_x = (SO3Scalar::hat(omega)).exp();
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
  cerr << "Test SO3" << endl << endl;
  cerr << "Double tests: " << endl;
  bool failed = so3explog_tests<double>();
  failed = failed || so3bracket_tests<double>();
  if (failed) {
    cerr << "failed!" << endl << endl;
    exit(-1);
  } else {
    cerr << "passed." << endl << endl;
  }

  cerr << "Float tests: " << endl;
  failed = failed || so3explog_tests<float>();
  failed = failed || so3bracket_tests<float>();
  if (failed) {
    cerr << "failed!" << endl << endl;
    exit(-1);
  } else {
    cerr << "passed." << endl << endl;
  }
  return 0;
}
