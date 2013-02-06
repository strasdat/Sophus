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
#include "se3.h"

using namespace Sophus;
using namespace std;

template<class Scalar>
bool se3explog_tests() {
  typedef SO3Group<Scalar> SO3Scalar;
  typedef SE3Group<Scalar> SE3Scalar;
  typedef Matrix<Scalar,3,1> Vector3Scalar;
  typedef Matrix<Scalar,4,4> Matrix4Scalar;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();
  const Scalar PI = SophusConstants<Scalar>::pi();


  vector<SE3Scalar> omegas;
  omegas.push_back(SE3Scalar(SO3Scalar::exp(Vector3Scalar(0.2, 0.5, 0.0)),
                             Vector3Scalar(0,0,0)));
  omegas.push_back(SE3Scalar(SO3Scalar::exp(Vector3Scalar(0.2, 0.5, -1.0)),
                             Vector3Scalar(10,0,0)));
  omegas.push_back(SE3Scalar(SO3Scalar::exp(Vector3Scalar(0., 0., 0.)),
                             Vector3Scalar(0,100,5)));
  omegas.push_back(SE3Scalar(SO3Scalar::exp(Vector3Scalar(0., 0., 0.00001)),
                             Vector3Scalar(0,0,0)));
  omegas.push_back(SE3Scalar(SO3Scalar::exp(Vector3Scalar(0., 0., 0.00001)),
                             Vector3Scalar(0,-0.00000001,0.0000000001)));
  omegas.push_back(SE3Scalar(SO3Scalar::exp(Vector3Scalar(0., 0., 0.00001)),
                             Vector3Scalar(0.01,0,0)));
  omegas.push_back(SE3Scalar(SO3Scalar::exp(Vector3Scalar(PI, 0, 0)),
                             Vector3Scalar(4,-5,0)));
  omegas.push_back(SE3Scalar(SO3Scalar::exp(Vector3Scalar(0.2, 0.5, 0.0)),
                             Vector3Scalar(0,0,0))
                   *SE3Scalar(SO3Scalar::exp(Vector3Scalar(PI, 0, 0)),
                              Vector3Scalar(0,0,0))
                   *SE3Scalar(SO3Scalar::exp(Vector3Scalar(-0.2, -0.5, -0.0)),
                              Vector3Scalar(0,0,0)));
  omegas.push_back(SE3Scalar(SO3Scalar::exp(Vector3Scalar(0.3, 0.5, 0.1)),
                             Vector3Scalar(2,0,-7))
                   *SE3Scalar(SO3Scalar::exp(Vector3Scalar(PI, 0, 0)),
                              Vector3Scalar(0,0,0))
                   *SE3Scalar(SO3Scalar::exp(Vector3Scalar(-0.3, -0.5, -0.1)),
                              Vector3Scalar(0,6,0)));

  bool failed = false;

  for (size_t i=0; i<omegas.size(); ++i) {
    Matrix4Scalar R1 = omegas[i].matrix();
    Matrix4Scalar R2 = SE3Scalar::exp(omegas[i].log()).matrix();
    Matrix4Scalar DiffR = R1-R2;
    Scalar nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "SE3 - exp(log(SE3))" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<omegas.size(); ++i) {
    Vector3Scalar p(1,2,4);
    Matrix4Scalar T = omegas[i].matrix();
    Vector3Scalar res1 = omegas[i]*p;
    Vector3Scalar res2
        = T.template topLeftCorner<3,3>()*p + T.template topRightCorner<3,1>();

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
    Matrix4Scalar q = omegas[i].matrix();
    Matrix4Scalar inv_q = omegas[i].inverse().matrix();
    Matrix4Scalar res = q*inv_q ;
    Matrix4Scalar I;
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

  for (size_t i=0; i<omegas.size(); ++i) {
    for (size_t j=0; j<omegas.size(); ++j) {
      Matrix4Scalar mul_resmat = (omegas[i]*omegas[j]).matrix();
      SE3Scalar fastmul_res = omegas[i];
      fastmul_res.fastMultiply(omegas[j]);
      Matrix4Scalar diff =  mul_resmat-fastmul_res.matrix();
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
bool se3bracket_tests() {
  typedef SE3Group<Scalar> SE3Scalar;
  typedef Matrix<Scalar,6,1> Vector6Scalar;
  typedef Matrix<Scalar,4,4> Matrix4Scalar;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();

  bool failed = false;
  vector<Vector6Scalar> vecs;
  Vector6Scalar tmp;
  tmp << 0,0,0,0,0,0;
  vecs.push_back(tmp);
  tmp << 1,0,0,0,0,0;
  vecs.push_back(tmp);
  tmp << 0,1,0,1,0,0;
  vecs.push_back(tmp);
  tmp << 0,-5,10,0,0,0;
  vecs.push_back(tmp);
  tmp << -1,1,0,0,0,1;
  vecs.push_back(tmp);
  tmp << 20,-1,0,-1,1,0;
  vecs.push_back(tmp);
  tmp << 30,5,-1,20,-1,0;
  vecs.push_back(tmp);
  for (size_t i=0; i<vecs.size(); ++i) {
    Vector6Scalar resDiff = vecs[i] - SE3Scalar::vee(SE3Scalar::hat(vecs[i]));
    if (resDiff.norm()>SMALL_EPS) {
      cerr << "Hat-vee Test" << endl;
      cerr  << "Test case: " << i <<  endl;
      cerr << resDiff.transpose() << endl;
      cerr << endl;
      failed = true;
    }

    for (size_t j=0; j<vecs.size(); ++j) {
      Vector6Scalar res1 = SE3Scalar::lieBracket(vecs[i],vecs[j]);
      Matrix4Scalar hati = SE3Scalar::hat(vecs[i]);
      Matrix4Scalar hatj = SE3Scalar::hat(vecs[j]);

      Vector6Scalar res2 = SE3Scalar::vee(hati*hatj-hatj*hati);
      Vector6Scalar resDiff = res1-res2;
      if (resDiff.norm()>SMALL_EPS) {
        cerr << "SE3 Lie Bracket Test" << endl;
        cerr  << "Test case: " << i << ", " <<j<< endl;
        cerr << vecs[i].transpose() << endl;
        cerr << vecs[j].transpose() << endl;
        cerr << resDiff.transpose() << endl;
        cerr << endl;
        failed = true;
      }
    }

    Vector6Scalar omega = vecs[i];
    Matrix4Scalar exp_x = SE3Scalar::exp(omega).matrix();
    Matrix4Scalar expmap_hat_x = (SE3Scalar::hat(omega)).exp();
    Matrix4Scalar DiffR = exp_x-expmap_hat_x;
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
  cerr << "Test SE3" << endl << endl;

  cerr << "Double tests: " << endl;
  bool failed = se3explog_tests<double>();
  failed = failed || se3bracket_tests<double>();
  if (failed) {
    cerr << "failed!" << endl << endl;
    exit(-1);
  } else {
    cerr << "passed." << endl << endl;
  }

  cerr << "Float tests: " << endl;
  failed = failed || se3explog_tests<float>();
  failed = failed || se3bracket_tests<float>();
  if (failed) {
    cerr << "failed!" << endl << endl;
    exit(-1);
  } else {
    cerr << "passed." << endl << endl;
  }
  return 0;
}
