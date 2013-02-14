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

#include "so3.hpp"

using namespace Sophus;
using namespace std;

template<class Scalar>
bool so3explog_tests() {
  typedef SO3Group<Scalar> SO3Type;
  typedef typename SO3Group<Scalar>::QuaternionType QuaternionType;
  typedef typename SO3Group<Scalar>::Point Point;
  typedef typename SO3Group<Scalar>::Adjoint Adjoint;
  typedef typename SO3Group<Scalar>::Tangent Tangent;
  typedef typename SO3Group<Scalar>::Transformation Transformation;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();
  const Scalar PI = SophusConstants<Scalar>::pi();

  vector<SO3Type> so3_vec;
  so3_vec.push_back(SO3Type(QuaternionType(0.1e-11, 0., 1., 0.)));
  so3_vec.push_back(SO3Type(QuaternionType(-1,0.00001,0.0,0.0)));
  so3_vec.push_back(SO3Type::exp(Point(0.2, 0.5, 0.0)));
  so3_vec.push_back(SO3Type::exp(Point(0.2, 0.5, -1.0)));
  so3_vec.push_back(SO3Type::exp(Point(0., 0., 0.)));
  so3_vec.push_back(SO3Type::exp(Point(0., 0., 0.00001)));
  so3_vec.push_back(SO3Type::exp(Point(M_PI, 0, 0)));
  so3_vec.push_back(SO3Type::exp(Point(0.2, 0.5, 0.0))
                   *SO3Type::exp(Point(M_PI, 0, 0))
                   *SO3Type::exp(Point(-0.2, -0.5, -0.0)));
  so3_vec.push_back(SO3Type::exp(Point(0.3, 0.5, 0.1))
                   *SO3Type::exp(Point(M_PI, 0, 0))
                   *SO3Type::exp(Point(-0.3, -0.5, -0.1)));

  bool failed = false;

  for (size_t i=0; i<so3_vec.size(); ++i) {
    Transformation R1 = so3_vec[i].matrix();
    Scalar theta;
    Transformation R2
        = SO3Type::exp(SO3Type::logAndTheta(so3_vec[i],&theta)).matrix();

    Transformation DiffR = R1-R2;
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

  for (size_t i=0; i<so3_vec.size(); ++i) {
    Point p(1,2,4);
    Transformation sR = so3_vec[i].matrix();
    Point res1 = so3_vec[i]*p;
    Point res2 = sR*p;

    Scalar nrm = (res1-res2).norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "Transform vector" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (res1-res2) <<endl;
      cerr << endl;
      failed = true;
    }
  }

  for (size_t i=0; i<so3_vec.size(); ++i) {
    Transformation q = so3_vec[i].matrix();
    Transformation inv_q = so3_vec[i].inverse().matrix();
    Transformation res = q*inv_q ;
    Transformation I;
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
  for (size_t i=0; i<so3_vec.size(); ++i) {
    Transformation T = so3_vec[i].matrix();
    Adjoint Ad = so3_vec[i].Adj();
    Tangent x;
    x << 1,2,3;
    Transformation I;
    I.setIdentity();
    Tangent ad1 = Ad*x;
    Tangent ad2 = SO3Type::vee(T*SO3Type::hat(x)
                                     *so3_vec[i].inverse().matrix());
    Scalar nrm = (ad1-ad2).norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "Adjoint" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (ad1-ad2).transpose() <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<so3_vec.size(); ++i) {
    for (size_t j=0; j<so3_vec.size(); ++j) {
      Transformation mul_resmat = (so3_vec[i]*so3_vec[j]).matrix();
      Scalar fastmul_res_raw[SO3Type::num_parameters];
      Eigen::Map<SO3Type> fastmul_res(fastmul_res_raw);
      fastmul_res = so3_vec[i];
      fastmul_res.fastMultiply(so3_vec[j]);
      Transformation diff =  mul_resmat-fastmul_res.matrix();
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
bool so3bracket_tests() {
  typedef SO3Group<Scalar> SO3Type;
  typedef typename SO3Group<Scalar>::Tangent Tangent;
  typedef typename SO3Group<Scalar>::Transformation Transformation;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();

  bool failed = false;
  vector<Tangent> vecs;
  vecs.push_back(Tangent(0,0,0));
  vecs.push_back(Tangent(1,0,0));
  vecs.push_back(Tangent(0,1,0));
  vecs.push_back(Tangent(M_PI_2,M_PI_2,0.0));
  vecs.push_back(Tangent(-1,1,0));
  vecs.push_back(Tangent(20,-1,0));
  vecs.push_back(Tangent(30,5,-1));
  for (unsigned int i=0; i<vecs.size(); ++i) {
    for (unsigned int j=0; j<vecs.size(); ++j) {
      Tangent res1 = SO3Type::lieBracket(vecs[i],vecs[j]);
      Transformation mat =
          SO3Type::hat(vecs[i])*SO3Type::hat(vecs[j])
          -SO3Type::hat(vecs[j])*SO3Type::hat(vecs[i]);
      Tangent res2 = SO3Type::vee(mat);
      Tangent resDiff = res1-res2;
      if (resDiff.norm()>SMALL_EPS) {
        cerr << "SO3 Lie Bracket Test" << endl;
        cerr  << "Test case: " << i << ", " <<j<< endl;
        cerr << res1-res2 << endl;
        cerr << endl;
        failed = true;
      }
    }

    Tangent omega = vecs[i];
    Transformation exp_x = SO3Type::exp(omega).matrix();
    Transformation expmap_hat_x = (SO3Type::hat(omega)).exp();
    Transformation DiffR = exp_x-expmap_hat_x;
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
