#include <iostream>
#include <vector>

#include <unsupported/Eigen/MatrixFunctions>
#include "se3.h"

using namespace Sophus;
using namespace std;

bool se3explog_tests()
{
  double pi = 3.14159265;
  vector<SE3> omegas;
  omegas.push_back(SE3(SO3::exp(Vector3d(0.2, 0.5, 0.0)),Vector3d(0,0,0)));
  omegas.push_back(SE3(SO3::exp(Vector3d(0.2, 0.5, -1.0)),Vector3d(10,0,0)));
  omegas.push_back(SE3(SO3::exp(Vector3d(0., 0., 0.)),Vector3d(0,100,5)));
  omegas.push_back(SE3(SO3::exp(Vector3d(0., 0., 0.00001)),Vector3d(0,0,0)));
  omegas.push_back(SE3(SO3::exp(Vector3d(0., 0., 0.00001)),Vector3d(0,-0.00000001,0.0000000001)));
  omegas.push_back(SE3(SO3::exp(Vector3d(0., 0., 0.00001)),Vector3d(0.01,0,0)));
  omegas.push_back(SE3(SO3::exp(Vector3d(pi, 0, 0)),Vector3d(4,-5,0)));
  omegas.push_back(SE3(SO3::exp(Vector3d(0.2, 0.5, 0.0)),Vector3d(0,0,0))
                   *SE3(SO3::exp(Vector3d(pi, 0, 0)),Vector3d(0,0,0))
                   *SE3(SO3::exp(Vector3d(-0.2, -0.5, -0.0)),Vector3d(0,0,0)));
  omegas.push_back(SE3(SO3::exp(Vector3d(0.3, 0.5, 0.1)),Vector3d(2,0,-7))
                   *SE3(SO3::exp(Vector3d(pi, 0, 0)),Vector3d(0,0,0))
                   *SE3(SO3::exp(Vector3d(-0.3, -0.5, -0.1)),Vector3d(0,6,0)));

  bool failed = false;

  for (size_t i=0; i<omegas.size(); ++i)
  {
    Matrix4d R1 = omegas[i].matrix();
    Matrix4d R2 = SE3::exp(omegas[i].log()).matrix();
    Matrix4d DiffR = R1-R2;
    double nrm = DiffR.norm();

    if (isnan(nrm) || nrm>SMALL_EPS)
    {
      cerr << "SE3 - exp(log(SE3))" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << DiffR <<endl;
      cerr << endl;
      failed = true;
    }
  }
  for (size_t i=0; i<omegas.size(); ++i)
  {
    Vector3d p(1,2,4);
    Matrix4d T = omegas[i].matrix();
    Vector3d res1 = omegas[i]*p;
    Vector3d res2 = T.topLeftCorner<3,3>()*p + T.topRightCorner<3,1>();

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
    Matrix4d q = omegas[i].matrix();
    Matrix4d inv_q = omegas[i].inverse().matrix();
    Matrix4d res = q*inv_q ;
    Matrix4d I;
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


bool se3bracket_tests()
{
  bool failed = false;
  vector<Vector6d> vecs;
  Vector6d tmp;
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
  for (size_t i=0; i<vecs.size(); ++i)
  {
    Vector6d resDiff = vecs[i] - SE3::vee(SE3::hat(vecs[i]));
    if (resDiff.norm()>SMALL_EPS)
    {
      cerr << "Hat-vee Test" << endl;
      cerr  << "Test case: " << i <<  endl;
      cerr << resDiff.transpose() << endl;
      cerr << endl;
      failed = true;
    }

    for (size_t j=0; j<vecs.size(); ++j)
    {
      Vector6d res1 = SE3::lieBracket(vecs[i],vecs[j]);
      Matrix4d hati = SE3::hat(vecs[i]);
      Matrix4d hatj = SE3::hat(vecs[j]);

      Vector6d res2 = SE3::vee(hati*hatj-hatj*hati);
      Vector6d resDiff = res1-res2;
      if (resDiff.norm()>SMALL_EPS)
      {
        cerr << "SE3 Lie Bracket Test" << endl;
        cerr  << "Test case: " << i << ", " <<j<< endl;
        cerr << vecs[i].transpose() << endl;
        cerr << vecs[j].transpose() << endl;
        cerr << resDiff.transpose() << endl;
        cerr << endl;
        failed = true;
      }
    }


    Vector6d omega = vecs[i];
    Matrix4d exp_x = SE3::exp(omega).matrix();
    Matrix4d expmap_hat_x = (SE3::hat(omega)).exp();
    Matrix4d DiffR = exp_x-expmap_hat_x;
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
  bool failed = se3explog_tests();
  failed = failed || se3bracket_tests();

  if (failed)
  {
    cerr << "failed" << endl;
    exit(-1);
  }
  return 0;
}
