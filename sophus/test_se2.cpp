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

using namespace Sophus;
using namespace std;

template<class Scalar>
bool se2explog_tests() {
  typedef SO2Group<Scalar> SO2Type;
  typedef SE2Group<Scalar> SE2Type;
  typedef typename SE2Group<Scalar>::Point Point;
  typedef typename SE2Group<Scalar>::Tangent Tangent;
  typedef typename SE2Group<Scalar>::Adjoint Adjoint;
  typedef typename SE2Group<Scalar>::Transformation Transformation;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();
  const Scalar PI = SophusConstants<Scalar>::pi();

  vector<SE2Type> se2_vec;
  se2_vec.push_back(SE2Type(SO2Type(0.0),Point(0,0)));
  se2_vec.push_back(SE2Type(SO2Type(0.2),Point(10,0)));
  se2_vec.push_back(SE2Type(SO2Type(0.),Point(0,100)));
  se2_vec.push_back(SE2Type(SO2Type(-1.),Point(20,-1)));
  se2_vec.push_back(SE2Type(SO2Type(0.00001),
                            Point(-0.00000001,0.0000000001)));
  se2_vec.push_back(SE2Type(SO2Type(0.2),Point(0,0))
                   *SE2Type(SO2Type(PI),Point(0,0))
                   *SE2Type(SO2Type(-0.2),Point(0,0)));
  se2_vec.push_back(SE2Type(SO2Type(0.3),Point(2,0))
                   *SE2Type(SO2Type(PI),Point(0,0))
                   *SE2Type(SO2Type(-0.3),Point(0,6)));

  bool failed = false;

  for (size_t i=0; i<se2_vec.size(); ++i) {
    Transformation R1 = se2_vec[i].matrix();
    Transformation R2 = SE2Type::exp(se2_vec[i].log()).matrix();
    Transformation DiffR = R1-R2;
    Scalar nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "SE2Type - exp(log(SE2Type))" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<se2_vec.size(); ++i) {
    Point p(1,2);
    Transformation T = se2_vec[i].matrix();
    Point res1 = se2_vec[i]*p;
    Point res2
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
  for (size_t i=0; i<se2_vec.size(); ++i) {
    Transformation T = se2_vec[i].matrix();
    Adjoint Ad = se2_vec[i].Adj();
    Tangent x;
    x << 1,2,1;
    Transformation I;
    I.setIdentity();
    Tangent ad1 = Ad*x;
    Tangent ad2 = SE2Type::vee(T*SE2Type::hat(x)
                                     *se2_vec[i].inverse().matrix());
    Scalar nrm = (ad1-ad2).norm();

    if (isnan(nrm) || nrm>SMALL_EPS) {
      cerr << "Adjoint" << endl;
      cerr  << "Test case: " << i << endl;
      cerr  << "Test case: " << i << endl;
      cerr << Ad <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<se2_vec.size(); ++i) {
    Transformation q = se2_vec[i].matrix();
    Transformation inv_q = se2_vec[i].inverse().matrix();
    Transformation res = q*inv_q ;
    Transformation I;
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

  for (size_t i=0; i<se2_vec.size(); ++i) {
    for (size_t j=0; j<se2_vec.size(); ++j) {
      Transformation mul_resmat = (se2_vec[i]*se2_vec[j]).matrix();
      Scalar fastmul_res_raw[SE2Type::num_parameters];
      Eigen::Map<SE2Type> fastmul_res(fastmul_res_raw);
      fastmul_res = se2_vec[i];
      fastmul_res.fastMultiply(se2_vec[j]);
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
bool se2bracket_tests() {
  typedef SE2Group<Scalar> SE2Type;
  typedef typename SE2Group<Scalar>::Tangent Tangent;
  typedef typename SE2Group<Scalar>::Adjoint Adjoint;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();

  bool failed = false;
  vector<Tangent> vecs;
  Tangent tmp;
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
    Tangent resDiff = vecs[i] - SE2Type::vee(SE2Type::hat(vecs[i]));
    if (resDiff.norm()>SMALL_EPS)
    {
      cerr << "Hat-vee Test" << endl;
      cerr  << "Test case: " << i <<  endl;
      cerr << resDiff.transpose() << endl;
      cerr << endl;
    }

    for (size_t j=0; j<vecs.size(); ++j) {
      Tangent res1 = SE2Type::lieBracket(vecs[i],vecs[j]);
      Adjoint hati = SE2Type::hat(vecs[i]);
      Adjoint hatj = SE2Type::hat(vecs[j]);

      Tangent res2 = SE2Type::vee(hati*hatj-hatj*hati);
      Tangent resDiff = res1-res2;
      if (resDiff.norm()>SMALL_EPS) {
        cerr << "SE2Type Lie Bracket Test" << endl;
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

    Tangent omega = vecs[i];
    Adjoint exp_x = SE2Type::exp(omega).matrix();
    Adjoint expmap_hat_x = (SE2Type::hat(omega)).exp();
    Adjoint DiffR = exp_x-expmap_hat_x;
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
