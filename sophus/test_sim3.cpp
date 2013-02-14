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

#include "sim3.hpp"

using namespace Sophus;
using namespace std;

template<class Scalar>
bool sim3explog_tests() {
  typedef RxSO3Group<Scalar> RxSO3Type;
  typedef Sim3Group<Scalar> Sim3Type;
  typedef Matrix<Scalar,4,1> Vector4Type;
  typedef typename Sim3Group<Scalar>::Point Point;
  typedef typename Sim3Group<Scalar>::Tangent Tangent;
  typedef typename Sim3Group<Scalar>::Transformation Transformation;
  typedef typename Sim3Group<Scalar>::Adjoint Adjoint;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();
  const Scalar PI = SophusConstants<Scalar>::pi();

  vector<Sim3Type> sim3_vec;
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0.2, 0.5, 0.0,1.)),
                              Point(0,0,0)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0.2, 0.5, -1.0,1.1)),
                              Point(10,0,0)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0.,1.1)),
                              Point(0,10,5)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0.00001, 0.)),
                              Point(0,0,0)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(
                                Vector4Type(0., 0., 0.00001, 0.0000001)),
                              Point(1,-1.00000001,2.0000000001)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0., 0., 0.00001, 0)),
                              Point(0.01,0,0)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(PI, 0, 0,0.9)),
                              Point(4,-5,0)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0.2, 0.5, 0.0,0)),
                              Point(0,0,0))
                     *Sim3Type(RxSO3Type::exp(Vector4Type(PI, 0, 0,0)),
                               Point(0,0,0))
                     *Sim3Type(RxSO3Type::exp(Vector4Type(-0.2, -0.5, -0.0,0)),
                               Point(0,0,0)));
  sim3_vec.push_back(Sim3Type(RxSO3Type::exp(Vector4Type(0.3, 0.5, 0.1,0)),
                              Point(2,0,-7))
                     *Sim3Type(RxSO3Type::exp(Vector4Type(PI, 0, 0,0)),
                               Point(0,0,0))
                     *Sim3Type(RxSO3Type::exp(Vector4Type(-0.3, -0.5, -0.1,0)),
                               Point(0,6,0)));

  bool failed = false;

  for (size_t i=0; i<sim3_vec.size(); ++i) {
    Transformation R1 = sim3_vec[i].matrix();
    Transformation R2 = Sim3Type::exp(sim3_vec[i].log()).matrix();
    Transformation DiffR = R1-R2;
    Scalar nrm = DiffR.norm();

    // ToDO: Force Sim3Type to be more accurate!
    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "Sim3Type - exp(log(Sim3Type))" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<sim3_vec.size(); ++i) {
    Point p(1,2,4);
    Transformation T = sim3_vec[i].matrix();
    Point res1 = sim3_vec[i]*p;
    Point res2
        = T.template topLeftCorner<3,3>()*p
        + T.template topRightCorner<3,1>();

    Scalar nrm = (res1-res2).norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "Transform vector" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (res1-res2) <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<sim3_vec.size(); ++i) {
    Transformation q = sim3_vec[i].matrix();
    Transformation inv_q = sim3_vec[i].inverse().matrix();
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
  for (size_t i=0; i<sim3_vec.size(); ++i) {
    Transformation T = sim3_vec[i].matrix();
    Adjoint Ad = sim3_vec[i].Adj();
    Tangent x;
    x << 0.9, 2, 3, 1.2, 2, 3, 1.1;
    Transformation I;
    I.setIdentity();
    Tangent ad1 = Ad*x;
    Tangent ad2 = Sim3Type::vee(T*Sim3Type::hat(x)
                                    *sim3_vec[i].inverse().matrix());
    Scalar nrm = (ad1-ad2).norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "Adjoint" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (ad1-ad2).transpose() <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<sim3_vec.size(); ++i) {
    for (size_t j=0; j<sim3_vec.size(); ++j) {
      Transformation mul_resmat = (sim3_vec[i]*sim3_vec[j]).matrix();
      Scalar mul_res_raw[Sim3Type::num_parameters];
      Map<Sim3Type> mul_res(mul_res_raw);
      mul_res = sim3_vec[i];
      mul_res *= sim3_vec[j];
      Transformation diff =  mul_resmat-mul_res.matrix();
      Scalar nrm = diff.norm();
      if (isnan(nrm) || nrm>SMALL_EPS) {
        cerr << diff <<endl;
        cerr << endl;
        failed = true;
      }
    }
  }
  return failed;
}

template<class Scalar>
bool sim3bracket_tests() {
  typedef RxSO3Group<Scalar> RxSO3Type;
  typedef Sim3Group<Scalar> Sim3Type;
  typedef Matrix<Scalar,4,1> Vector4Type;
  typedef typename Sim3Group<Scalar>::Point Point;
  typedef typename Sim3Group<Scalar>::Tangent Tangent;
  typedef typename Sim3Group<Scalar>::Transformation Transformation;
  typedef typename Sim3Group<Scalar>::Adjoint Adjoint;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();

  bool failed = false;
  vector<Tangent> vecs;
  Tangent tmp;
  tmp << 0,0,0,0,0,0,0;
  vecs.push_back(tmp);
  tmp << 1,0,0,0,0,0,0;
  vecs.push_back(tmp);
  tmp << 0,1,0,1,0,0,0.1;
  vecs.push_back(tmp);
  tmp << 0,0,1,0,1,0,0.1;
  vecs.push_back(tmp);
  tmp << -1,1,0,0,0,1,-0.1;
  vecs.push_back(tmp);
  tmp << 20,-1,0,-1,1,0,-0.1;
  vecs.push_back(tmp);
  tmp << 30,5,-1,20,-1,0,1.5;
  vecs.push_back(tmp);
  for (size_t i=0; i<vecs.size(); ++i) {
    Tangent resDiff = vecs[i] - Sim3Type::vee(Sim3Type::hat(vecs[i]));
    if (resDiff.norm()>SMALL_EPS) {
      cerr << "Hat-vee Test" << endl;
      cerr  << "Test case: " << i <<  endl;
      cerr << resDiff.transpose() << endl;
      cerr << endl;
      failed = true;
    }

    for (size_t j=0; j<vecs.size(); ++j) {
      Tangent res1 = Sim3Type::lieBracket(vecs[i],vecs[j]);
      Transformation hati = Sim3Type::hat(vecs[i]);
      Transformation hatj = Sim3Type::hat(vecs[j]);

      Tangent res2 = Sim3Type::vee(hati*hatj-hatj*hati);
      Tangent resDiff = res1-res2;
      if (resDiff.norm()>SMALL_EPS) {
        cerr << "Sim3Type Lie Bracket Test" << endl;
        cerr  << "Test case: " << i << ", " <<j<< endl;
        cerr << vecs[i].transpose() << endl;
        cerr << vecs[j].transpose() << endl;
        cerr << resDiff.transpose() << endl;
        cerr << endl;
        failed = true;
      }
    }

    Tangent omega = vecs[i];
    Transformation exp_x = Sim3Type::exp(omega).matrix();
    Transformation expmap_hat_x = (Sim3Type::hat(omega)).exp();
    Transformation DiffR = exp_x-expmap_hat_x;
    Scalar nrm = DiffR.norm();

    if (isnan(nrm) || nrm>static_cast<Scalar>(10)*SMALL_EPS) {
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
  cerr << "Test Sim3" << endl << endl;

  cerr << "Double tests: " << endl;
  bool failed = sim3explog_tests<double>();
  failed = failed || sim3bracket_tests<double>();
  if (failed) {
    cerr << "failed!" << endl << endl;
    exit(-1);
  } else {
    cerr << "passed." << endl << endl;
  }

  cerr << "Float tests: " << endl;
  failed = failed || sim3explog_tests<float>();
  failed = failed || sim3bracket_tests<float>();
  if (failed) {
    cerr << "failed!" << endl << endl;
    exit(-1);
  } else {
    cerr << "passed." << endl << endl;
  }
  return 0;
}

