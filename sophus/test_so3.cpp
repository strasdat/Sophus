#include <iostream>
#include <vector>


#include <unsupported/Eigen/MatrixFunctions>

#include "so3.h"

using namespace Sophus;
using namespace std;


bool so3explog_tests()
{

  double pi = 3.14159265;
  vector<SO3> omegas;
  omegas.push_back(SO3::exp(Vector3d(0.2, 0.5, 0.0)));
  omegas.push_back(SO3::exp(Vector3d(0.2, 0.5, -1.0)));
  omegas.push_back(SO3::exp(Vector3d(0., 0., 0.)));
  omegas.push_back(SO3::exp(Vector3d(0., 0., 0.00001)));
  omegas.push_back(SO3::exp(Vector3d(pi, 0, 0)));
  omegas.push_back(SO3::exp(Vector3d(0.2, 0.5, 0.0))
                   *SO3::exp(Vector3d(pi, 0, 0))
                   *SO3::exp(Vector3d(-0.2, -0.5, -0.0)));
  omegas.push_back(SO3::exp(Vector3d(0.3, 0.5, 0.1))
                   *SO3::exp(Vector3d(pi, 0, 0))
                   *SO3::exp(Vector3d(-0.3, -0.5, -0.1)));

  bool failed = false;

  for (size_t i=0; i<omegas.size(); ++i)
  {
    Matrix3d R1 = omegas[i].matrix();
    Matrix3d R2 = SO3::exp(omegas[i].log()).matrix();

    Matrix3d DiffR = R1-R2;
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

  for (size_t i=0; i<omegas.size(); ++i)
  {
    Vector3d p(1,2,4);
    Matrix3d sR = omegas[i].matrix();
    Vector3d res1 = omegas[i]*p;
    Vector3d res2 = sR*p;

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


bool so3bracket_tests()
{
  bool failed = false;
  vector<Vector3d> vecs;
  vecs.push_back(Vector3d(0,0,0));
  vecs.push_back(Vector3d(1,0,0));
  vecs.push_back(Vector3d(0,1,0));
  vecs.push_back(Vector3d(0,0,1));
  vecs.push_back(Vector3d(-1,1,0));
  vecs.push_back(Vector3d(20,-1,0));
  vecs.push_back(Vector3d(30,5,-1));
  for (uint i=0; i<vecs.size(); ++i)
  {
    for (uint j=0; j<vecs.size(); ++j)
    {
      Vector3d res1 = SO3::lieBracket(vecs[i],vecs[j]);
      Matrix3d mat =
          SO3::hat(vecs[i])*SO3::hat(vecs[j])
          -SO3::hat(vecs[j])*SO3::hat(vecs[i]);
      Vector3d res2 = SO3::vee(mat);
      Vector3d resDiff = res1-res2;
      if (resDiff.norm()>SMALL_EPS)
      {
        cerr << "SO3 Lie Bracket Test" << endl;
        cerr  << "Test case: " << i << ", " <<j<< endl;
        cerr << res1-res2 << endl;
        cerr << endl;
        failed = true;
      }
    }

    Vector3d omega = vecs[i];
    Matrix3d exp_x = SO3::exp(omega).matrix();
    Matrix3d expmap_hat_x = (SO3::hat(omega)).exp();
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
  bool failed = so3explog_tests();
  failed = failed || so3bracket_tests();

  if (failed)
  {
    cerr << "failed" << endl;
    exit(-1);
  }
  return 0;
}
