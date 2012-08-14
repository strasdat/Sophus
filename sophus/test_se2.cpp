#include <iostream>
#include <vector>

#include <unsupported/Eigen/MatrixFunctions>
#include "se2.h"
#include "so3.h"

using namespace Sophus;
using namespace std;

bool se2explog_tests()
{
  double pi = 3.14159265;
  vector<SE2> omegas;
  omegas.push_back(SE2(SO2(0.0),Vector2d(0,0)));
  omegas.push_back(SE2(SO2(0.2),Vector2d(10,0)));
  omegas.push_back(SE2(SO2(0.),Vector2d(0,100)));
  omegas.push_back(SE2(SO2(-1.),Vector2d(20,-1)));
  omegas.push_back(SE2(SO2(0.00001),Vector2d(-0.00000001,0.0000000001)));
  omegas.push_back(SE2(SO2(0.2),Vector2d(0,0))
                   *SE2(SO2(pi),Vector2d(0,0))
                   *SE2(SO2(-0.2),Vector2d(0,0)));
  omegas.push_back(SE2(SO2(0.3),Vector2d(2,0))
                   *SE2(SO2(pi),Vector2d(0,0))
                   *SE2(SO2(-0.3),Vector2d(0,6)));

  bool failed = false;

  for (size_t i=0; i<omegas.size(); ++i)
  {
    Matrix3d R1 = omegas[i].matrix();
    Matrix3d R2 = SE2::exp(omegas[i].log()).matrix();
    Matrix3d DiffR = R1-R2;
    double nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS)
    {
      cerr << "SE2 - exp(log(SE2))" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<omegas.size(); ++i)
  {
    Vector2d p(1,2);
    Matrix3d T = omegas[i].matrix();
    Vector2d res1 = omegas[i]*p;
    Vector2d res2 = T.topLeftCorner<2,2>()*p + T.topRightCorner<2,1>();

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

  for (size_t i=0; i<omegas.size(); ++i)
  {
    Matrix3d q = omegas[i].matrix();
    Matrix3d inv_q = omegas[i].inverse().matrix();
    Matrix3d res = q*inv_q ;
    Matrix3d I;
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
  return failed;

}


bool se2bracket_tests()
{
  bool failed = false;
  vector<Vector3d> vecs;
  Vector3d tmp;
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
  for (size_t i=0; i<vecs.size(); ++i)
  {
    Vector3d resDiff = vecs[i] - SE2::vee(SE2::hat(vecs[i]));
    if (resDiff.norm()>SMALL_EPS)
    {
      cerr << "Hat-vee Test" << endl;
      cerr  << "Test case: " << i <<  endl;
      cerr << resDiff.transpose() << endl;
      cerr << endl;
    }

    for (size_t j=0; j<vecs.size(); ++j)
    {
      Vector3d res1 = SE2::lieBracket(vecs[i],vecs[j]);
      Matrix3d hati = SE2::hat(vecs[i]);
      Matrix3d hatj = SE2::hat(vecs[j]);

      Vector3d res2 = SE2::vee(hati*hatj-hatj*hati);
      Vector3d resDiff = res1-res2;
      if (resDiff.norm()>SMALL_EPS)
      {
        cerr << "SE2 Lie Bracket Test" << endl;
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


    Vector3d omega = vecs[i];
    Matrix3d exp_x = SE2::exp(omega).matrix();
    Matrix3d expmap_hat_x = (SE2::hat(omega)).exp();
    Matrix3d DiffR = exp_x-expmap_hat_x;
    double nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS)
    {
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



int main()
{
  bool failed = se2explog_tests();
  failed = failed || se2bracket_tests();

  if (failed)
  {
    cerr << "failed" << endl;
    exit(-1);
  }
  return 0;
}
