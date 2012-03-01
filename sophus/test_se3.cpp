#include <iostream>
#include <vector>

#include "se3.h"

using namespace Sophus;
using namespace std;

void se3explog_tests()
{
  double pi = 3.14159265;
  vector<SE3> omegas;
  omegas.push_back(SE3(SO3::exp(Vector3d(0.2, 0.5, 0.0)),Vector3d(0,0,0)));
  omegas.push_back(SE3(SO3::exp(Vector3d(0.2, 0.5, -1.0)),Vector3d(10,0,0)));
  omegas.push_back(SE3(SO3::exp(Vector3d(0., 0., 0.)),Vector3d(0,100,5)));
  omegas.push_back(SE3(SO3::exp(Vector3d(0., 0., 0.00001)),Vector3d(0,0,0)));
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
  if (failed)
    exit(-1);
}


//void so3bracket_tests()
//{
//  vector<Vector3d> vecs;
//  vecs.push_back(Vector3d(0,0,0));
//  vecs.push_back(Vector3d(1,0,0));
//  vecs.push_back(Vector3d(0,1,0));
//  vecs.push_back(Vector3d(0,0,1));
//  vecs.push_back(Vector3d(-1,1,0));
//  vecs.push_back(Vector3d(20,-1,0));
//  vecs.push_back(Vector3d(30,5,-1));
//  for (uint i=0; i<vecs.size(); ++i)
//  {
//    for (uint j=0; j<vecs.size(); ++j)
//    {
//      Vector3d res1 = SO3::lieBracket(vecs[i],vecs[j]);
//      Matrix3d mat = SO3::hat(vecs[i])*SO3::hat(vecs[j])-SO3::hat(vecs[j])*SO3::hat(vecs[i]);
//      Vector3d res2 = SO3::vee(mat);
//      Vector3d resDiff = res1-res2;
//      if (resDiff.norm()>SMALL_EPS)
//      {
//        cerr << "SO3 Lie Bracket Test" << endl;
//        cerr  << "Test case: " << i << ", " <<j<< endl;
//        cerr << res1-res2 << endl;
//        cerr << endl;
//      }
//    }
//  }
//}



int main()
{
  se3explog_tests();
  //so3bracket_tests();
  return 0;
}
