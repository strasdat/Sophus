#include <iostream>
#include <vector>

#include <unsupported/Eigen/MatrixFunctions>

#include "so2.h"
#include "so3.h"

using namespace Sophus;
using namespace std;

bool so2explog_tests()
{
  double pi = 3.14159265;
  vector<SO2> so2;
  so2.push_back(SO2::exp(0.0));
  so2.push_back(SO2::exp(0.2));
  so2.push_back(SO2::exp(10.));
  so2.push_back(SO2::exp(0.00001));
  so2.push_back(SO2::exp(pi));
  so2.push_back(SO2::exp(0.2)
                   *SO2::exp(pi)
                   *SO2::exp(-0.2));
  so2.push_back(SO2::exp(-0.3)
                   *SO2::exp(pi)
                   *SO2::exp(0.3));


  bool failed = false;

  for (size_t i=0; i<so2.size(); ++i)
  {
    Matrix2d R1 = so2[i].matrix();
    Matrix2d R2 = SO2::exp(so2[i].log()).matrix();

    Matrix2d DiffR = R1-R2;
    double nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS)
    {
      cerr << "SO3 - exp(log(SO3))" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }

  for (size_t i=0; i<so2.size(); ++i)
  {
    Vector2d p(1,2);
    Matrix2d R = so2[i].matrix();
    Vector2d res1 = so2[i]*p;
    Vector2d res2 = R*p;

    double nrm = (res1-res2).norm();

    if (isnan(nrm) || nrm>SMALL_EPS)
    {
      cerr << "Transform vector" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (res1-res2) <<endl;
      cerr << endl;
      failed = true;
    }
  }

  for (size_t i=0; i<so2.size(); ++i)
  {
    Matrix2d q = so2[i].matrix();
    Matrix2d inv_q = so2[i].inverse().matrix();
    Matrix2d res = q*inv_q ;
    Matrix2d I;
    I.setIdentity();

    double nrm = (res-I).norm();

    if (isnan(nrm) || nrm>SMALL_EPS)
    {
      cerr << "Inverse" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << (res-I) <<endl;
      cerr << endl;
      failed = true;
    }
  }

  for (size_t i=0; i<so2.size(); ++i)
  {
    double omega = so2[i].log();
    Matrix2d exp_x = SO2::exp(omega).matrix();
    Matrix2d expmap_hat_x = (SO2::hat(omega)).exp();
    Matrix2d DiffR = exp_x-expmap_hat_x;
    double nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS)
    {
      cerr << "expmap(hat(x)) - exp(x)" << endl;
      cerr  << "Test case: " << i << endl;
//      cerr << exp_x <<endl;
//      cerr << expmap_hat_x <<endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }
  return failed;
}





int main()
{
  if (so2explog_tests())
  {
    exit(-1);
  }
  return 0;
}
