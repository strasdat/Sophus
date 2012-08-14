#include <iostream>
#include <vector>

#include <unsupported/Eigen/MatrixFunctions>
#include "scso3.h"

using namespace Sophus;
using namespace std;


bool scso3explog_tests()
{
  double pi = 3.14159265;
  vector<ScSO3> omegas;
  omegas.push_back(ScSO3::exp(Vector4d(0.2, 0.5, 0.0, 1.)));
  omegas.push_back(ScSO3::exp(Vector4d(0.2, 0.5, -1.0, 1.1)));
  omegas.push_back(ScSO3::exp(Vector4d(0., 0., 0., 1.1)));
  omegas.push_back(ScSO3::exp(Vector4d(0., 0., 0.00001, 0.)));
  omegas.push_back(ScSO3::exp(Vector4d(0., 0., 0.00001, 0.00001)));
  omegas.push_back(ScSO3::exp(Vector4d(0., 0., 0.00001, 0)));
  omegas.push_back(ScSO3::exp(Vector4d(pi, 0, 0, 0.9)));
  omegas.push_back(ScSO3::exp(Vector4d(0.2, 0.5, 0.0,0))
                   *ScSO3::exp(Vector4d(pi, 0, 0,0.0))
                   *ScSO3::exp(Vector4d(-0.2, -0.5, -0.0,0)));
  omegas.push_back(ScSO3::exp(Vector4d(0.3, 0.5, 0.1,0))
                   *ScSO3::exp(Vector4d(pi, 0, 0,0))
                   *ScSO3::exp(Vector4d(-0.3, -0.5, -0.1,0)));

  bool failed = false;

  for (size_t i=0; i<omegas.size(); ++i)
  {
    Matrix3d sR1 = omegas[i].matrix();
    Matrix3d sR2 = ScSO3::exp(omegas[i].log()).matrix();
    Matrix3d DiffR = sR1-sR2;
    double nrm = DiffR.norm();

    //// ToDO: Force ScSO3 to be more accurate!
    if (isnan(nrm) || nrm>SMALL_EPS)
    {
      cerr << "ScSO3 - exp(log(ScSO3))" << endl;
      cerr  << "Test case: " << i << endl;
      cerr << sR1 << endl;
      cerr << omegas[i].log() << endl;
      cerr << sR2 << endl;
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
    Matrix3d R = omegas[i].rotationMatrix();
    cerr << R*R.transpose() << endl;

  }
  return failed;
}


bool scso3bracket_tests()
{
  bool failed = false;
  vector<Vector4d> vecs;
  Vector4d tmp;
  tmp << 0,0,0,0;
  vecs.push_back(tmp);
  tmp << 1,0,0,0;
  vecs.push_back(tmp);
  tmp << 1,0,0,0.1;
  vecs.push_back(tmp);
  tmp << 0,1,0,0.1;
  vecs.push_back(tmp);
  tmp << 0,0,1,-0.1;
  vecs.push_back(tmp);
  tmp << -1,1,0,-0.1;
  vecs.push_back(tmp);
  tmp << 20,-1,0,2;
  vecs.push_back(tmp);
  for (size_t i=0; i<vecs.size(); ++i)
  {
    Vector4d resDiff = vecs[i] - ScSO3::vee(ScSO3::hat(vecs[i]));
    if (resDiff.norm()>SMALL_EPS)
    {
      cerr << "Hat-vee Test" << endl;
      cerr  << "Test case: " << i <<  endl;
      cerr << resDiff.transpose() << endl;
      cerr << endl;
    }

    for (size_t j=0; j<vecs.size(); ++j)
    {
      Vector4d res1 = ScSO3::lieBracket(vecs[i],vecs[j]);
      Matrix3d hati = ScSO3::hat(vecs[i]);
      Matrix3d hatj = ScSO3::hat(vecs[j]);

      Vector4d res2 = ScSO3::vee(hati*hatj-hatj*hati);
      Vector4d resDiff = res1-res2;
      if (resDiff.norm()>SMALL_EPS)
      {
        cerr << "ScSO3 Lie Bracket Test" << endl;
        cerr  << "Test case: " << i << ", " <<j<< endl;
        cerr << vecs[i].transpose() << endl;
        cerr << vecs[j].transpose() << endl;
        cerr << resDiff.transpose() << endl;
        cerr << endl;
        failed = true;
      }
    }



    Vector4d omega = vecs[i];
    Matrix3d exp_x = ScSO3::exp(omega).matrix();
    Matrix3d expmap_hat_x = (ScSO3::hat(omega)).exp();
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
  bool failed = scso3explog_tests();
  failed = failed || scso3bracket_tests();

  if (failed)
  {
    cerr << "failed" << endl;
    exit(-1);
  }
  return 0;
}

