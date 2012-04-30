#include <iostream>
#include <vector>

#include "sim3.h"

using namespace Sophus;
using namespace std;

void sim3explog_tests()
{
  double pi = 3.14159265;
  vector<Sim3> omegas;
  omegas.push_back(Sim3(ScSO3::exp(Vector4d(0.2, 0.5, 0.0,1.)),Vector3d(0,0,0)));
  omegas.push_back(Sim3(ScSO3::exp(Vector4d(0.2, 0.5, -1.0,1.1)),Vector3d(10,0,0)));
  omegas.push_back(Sim3(ScSO3::exp(Vector4d(0., 0., 0.,1.1)),Vector3d(0,100,5)));
  omegas.push_back(Sim3(ScSO3::exp(Vector4d(0., 0., 0.00001, 0.)),Vector3d(0,0,0)));
  omegas.push_back(Sim3(ScSO3::exp(Vector4d(0., 0., 0.00001, 0.0000001)),Vector3d(1,-1.00000001,2.0000000001)));
  omegas.push_back(Sim3(ScSO3::exp(Vector4d(0., 0., 0.00001, 0)),Vector3d(0.01,0,0)));
  omegas.push_back(Sim3(ScSO3::exp(Vector4d(pi, 0, 0,0.9)),Vector3d(4,-5,0)));
  omegas.push_back(Sim3(ScSO3::exp(Vector4d(0.2, 0.5, 0.0,0)),Vector3d(0,0,0))
                   *Sim3(ScSO3::exp(Vector4d(pi, 0, 0,0)),Vector3d(0,0,0))
                   *Sim3(ScSO3::exp(Vector4d(-0.2, -0.5, -0.0,0)),Vector3d(0,0,0)));
  omegas.push_back(Sim3(ScSO3::exp(Vector4d(0.3, 0.5, 0.1,0)),Vector3d(2,0,-7))
                   *Sim3(ScSO3::exp(Vector4d(pi, 0, 0,0)),Vector3d(0,0,0))
                   *Sim3(ScSO3::exp(Vector4d(-0.3, -0.5, -0.1,0)),Vector3d(0,6,0)));

  bool failed = false;

  for (size_t i=0; i<omegas.size(); ++i)
  {
    Matrix4d R1 = omegas[i].matrix();
    Matrix4d R2 = Sim3::exp(omegas[i].log()).matrix();
    Matrix4d DiffR = R1-R2;
    double nrm = DiffR.norm();

    // ToDO: Force Sim3 to be more accurate!
    if (isnan(nrm) || nrm>SMALL_EPS)
    {
      cerr << "Sim3 - exp(log(Sim3))" << endl;
      cerr  << "Test case: " << i << endl;
//      cerr << R1 <<endl;
//      cerr << omegas[i].log().transpose() <<endl;
//      cerr << R2 <<endl;
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
}


void sim3bracket_tests()
{
  vector<Vector7d> vecs;
  Vector7d tmp;
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
  tmp << 30,5,-1,20,-1,0,2;
  vecs.push_back(tmp);
  for (uint i=0; i<vecs.size(); ++i)
  {
    Vector7d resDiff = vecs[i] - Sim3::vee(Sim3::hat(vecs[i]));
    if (resDiff.norm()>SMALL_EPS)
    {
      cerr << "Hat-vee Test" << endl;
      cerr  << "Test case: " << i <<  endl;
      cerr << resDiff.transpose() << endl;
      cerr << endl;
    }

    for (uint j=0; j<vecs.size(); ++j)
    {
      Vector7d res1 = Sim3::lieBracket(vecs[i],vecs[j]);
      Matrix4d hati = Sim3::hat(vecs[i]);
      Matrix4d hatj = Sim3::hat(vecs[j]);

      Vector7d res2 = Sim3::vee(hati*hatj-hatj*hati);
      Vector7d resDiff = res1-res2;
      if (resDiff.norm()>SMALL_EPS)
      {
        cerr << "Sim3 Lie Bracket Test" << endl;
        cerr  << "Test case: " << i << ", " <<j<< endl;
        cerr << vecs[i].transpose() << endl;
        cerr << vecs[j].transpose() << endl;
        cerr << resDiff.transpose() << endl;
        cerr << endl;
      }
    }
  }
}



int main()
{
  sim3explog_tests();
  sim3bracket_tests();
  return 0;
}

