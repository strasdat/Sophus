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
  typedef typename SE2Group<Scalar>::PointType PointType;
  typedef typename SE2Group<Scalar>::TangentType TangentType;
  typedef typename SE2Group<Scalar>::AdjointType AdjointType;
  typedef typename SE2Group<Scalar>::TransformationType TransformationType;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();
  const Scalar PI = SophusConstants<Scalar>::pi();

  vector<SE2Type> se2_vec;
  se2_vec.push_back(SE2Type(SO2Type(0.0),PointType(0,0)));
  se2_vec.push_back(SE2Type(SO2Type(0.2),PointType(10,0)));
  se2_vec.push_back(SE2Type(SO2Type(0.),PointType(0,100)));
  se2_vec.push_back(SE2Type(SO2Type(-1.),PointType(20,-1)));
  se2_vec.push_back(SE2Type(SO2Type(0.00001),
                            PointType(-0.00000001,0.0000000001)));
  se2_vec.push_back(SE2Type(SO2Type(0.2),PointType(0,0))
                   *SE2Type(SO2Type(PI),PointType(0,0))
                   *SE2Type(SO2Type(-0.2),PointType(0,0)));
  se2_vec.push_back(SE2Type(SO2Type(0.3),PointType(2,0))
                   *SE2Type(SO2Type(PI),PointType(0,0))
                   *SE2Type(SO2Type(-0.3),PointType(0,6)));

  bool failed = false;

  for (size_t i=0; i<se2_vec.size(); ++i) {
    TransformationType R1 = se2_vec[i].matrix();
    TransformationType R2 = SE2Type::exp(se2_vec[i].log()).matrix();
    TransformationType DiffR = R1-R2;
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
    PointType p(1,2);
    TransformationType T = se2_vec[i].matrix();
    PointType res1 = se2_vec[i]*p;
    PointType res2
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
    TransformationType T = se2_vec[i].matrix();
    AdjointType Ad = se2_vec[i].Adj();
    TangentType x;
    x << 1,2,1;
    TransformationType I;
    I.setIdentity();
    TangentType ad1 = Ad*x;
    TangentType ad2 = SE2Type::vee(T*SE2Type::hat(x)
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
    TransformationType q = se2_vec[i].matrix();
    TransformationType inv_q = se2_vec[i].inverse().matrix();
    TransformationType res = q*inv_q ;
    TransformationType I;
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
      TransformationType mul_resmat = (se2_vec[i]*se2_vec[j]).matrix();
      Scalar fastmul_res_raw[SE2Type::num_parameters];
      Eigen::Map<SE2Type> fastmul_res(fastmul_res_raw);
      fastmul_res = se2_vec[i];
      fastmul_res.fastMultiply(se2_vec[j]);
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

template<class Scalar>
bool se2bracket_tests() {
  typedef SE2Group<Scalar> SE2Type;
  typedef typename SE2Group<Scalar>::TangentType TangentType;
  typedef typename SE2Group<Scalar>::AdjointType AdjointType;
  const Scalar SMALL_EPS = SophusConstants<Scalar>::epsilon();

  bool failed = false;
  vector<TangentType> vecs;
  TangentType tmp;
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
    TangentType resDiff = vecs[i] - SE2Type::vee(SE2Type::hat(vecs[i]));
    if (resDiff.norm()>SMALL_EPS)
    {
      cerr << "Hat-vee Test" << endl;
      cerr  << "Test case: " << i <<  endl;
      cerr << resDiff.transpose() << endl;
      cerr << endl;
    }

    for (size_t j=0; j<vecs.size(); ++j) {
      TangentType res1 = SE2Type::lieBracket(vecs[i],vecs[j]);
      AdjointType hati = SE2Type::hat(vecs[i]);
      AdjointType hatj = SE2Type::hat(vecs[j]);

      TangentType res2 = SE2Type::vee(hati*hatj-hatj*hati);
      TangentType resDiff = res1-res2;
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

    TangentType omega = vecs[i];
    AdjointType exp_x = SE2Type::exp(omega).matrix();
    AdjointType expmap_hat_x = (SE2Type::hat(omega)).exp();
    AdjointType DiffR = exp_x-expmap_hat_x;
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
