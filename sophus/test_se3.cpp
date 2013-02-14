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
#include "se3.hpp"

using namespace Sophus;
using namespace std;

template<class Scalar>
bool se3explog_tests() {
  typedef SO3Group<Scalar> SO3Type;
  typedef SE3Group<Scalar> SE3Type;
  typedef typename SE3Group<Scalar>::Point Point;
  typedef typename SE3Group<Scalar>::Tangent Tangent;
  typedef typename SE3Group<Scalar>::Transformation Transformation;
  typedef typename SE3Group<Scalar>::Adjoint Adjoint;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();
  const Scalar PI = SophusConstants<Scalar>::pi();


  vector<SE3Type> se3_vec;
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0.2, 0.5, 0.0)),
                             Point(0,0,0)));
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0.2, 0.5, -1.0)),
                             Point(10,0,0)));
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.)),
                             Point(0,100,5)));
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.00001)),
                             Point(0,0,0)));
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.00001)),
                             Point(0,-0.00000001,0.0000000001)));
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.00001)),
                             Point(0.01,0,0)));
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(PI, 0, 0)),
                             Point(4,-5,0)));
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0.2, 0.5, 0.0)),
                             Point(0,0,0))
                   *SE3Type(SO3Type::exp(Point(PI, 0, 0)),
                              Point(0,0,0))
                   *SE3Type(SO3Type::exp(Point(-0.2, -0.5, -0.0)),
                              Point(0,0,0)));
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0.3, 0.5, 0.1)),
                             Point(2,0,-7))
                   *SE3Type(SO3Type::exp(Point(PI, 0, 0)),
                              Point(0,0,0))
                   *SE3Type(SO3Type::exp(Point(-0.3, -0.5, -0.1)),
                              Point(0,6,0)));

  bool failed = false;

  for (size_t i=0; i<se3_vec.size(); ++i) {
    Transformation R1 = se3_vec[i].matrix();
    Transformation R2 = SE3Type::exp(se3_vec[i].log()).matrix();
    Transformation DiffR = R1-R2;
    Scalar nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "SE3 - exp(log(SE3))" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<se3_vec.size(); ++i) {
    Point p(1,2,4);
    Transformation T = se3_vec[i].matrix();
    Point res1 = se3_vec[i]*p;
    Point res2
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
  for (size_t i=0; i<se3_vec.size(); ++i) {
    Transformation T = se3_vec[i].matrix();
    Adjoint Ad = se3_vec[i].Adj();
    Tangent x;
    x << 1,2,1,2,1,2;
    Transformation I;
    I.setIdentity();
    Tangent ad1 = Ad*x;
    Tangent ad2 = SE3Type::vee(T*SE3Type::hat(x)
                                     *se3_vec[i].inverse().matrix());
    Scalar nrm = (ad1-ad2).norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "Adjoint" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (ad1-ad2).transpose() <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<se3_vec.size(); ++i) {
    for (size_t j=0; j<se3_vec.size(); ++j) {
      Transformation mul_resmat = (se3_vec[i]*se3_vec[j]).matrix();
      Scalar fastmul_res_raw[SE3Type::num_parameters];
      Eigen::Map<SE3Type> fastmul_res(fastmul_res_raw);
      fastmul_res = se3_vec[i];
      fastmul_res.fastMultiply(se3_vec[j]);
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
bool se3bracket_tests() {
  typedef SE3Group<Scalar> SE3Type;
  typedef typename SE3Group<Scalar>::Tangent Tangent;
  typedef typename SE3Group<Scalar>::Transformation Transformation;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();

  bool failed = false;
  vector<Tangent> vecs;
  Tangent tmp;
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
    Tangent resDiff = vecs[i] - SE3Type::vee(SE3Type::hat(vecs[i]));
    if (resDiff.norm()>SMALL_EPS) {
      cerr << "Hat-vee Test" << endl;
      cerr  << "Test case: " << i <<  endl;
      cerr << resDiff.transpose() << endl;
      cerr << endl;
      failed = true;
    }

    for (size_t j=0; j<vecs.size(); ++j) {
      Tangent res1 = SE3Type::lieBracket(vecs[i],vecs[j]);
      Transformation hati = SE3Type::hat(vecs[i]);
      Transformation hatj = SE3Type::hat(vecs[j]);

      Tangent res2 = SE3Type::vee(hati*hatj-hatj*hati);
      Tangent resDiff = res1-res2;
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

    Tangent omega = vecs[i];
    Transformation exp_x = SE3Type::exp(omega).matrix();
    Transformation expmap_hat_x = (SE3Type::hat(omega)).exp();
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
