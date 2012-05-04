// This file is part of Sophus.
//
// Copyright 2011 Hauke Strasdat (Imperial College London)
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

#include "sim3.h"

#include <iostream>

namespace Sophus
{
Sim3
::Sim3()
{
  translation_.setZero();
}

Sim3
::Sim3(const ScSO3& scso3, const Vector3d & t)
  :scso3_(scso3),translation_(t)
{
}

Sim3
::Sim3(const Sim3 & sim3)
  : scso3_(sim3.scso3_),translation_(sim3.translation_)
{
}

Sim3 Sim3
::from_SE3(const SE3 & se3)
{
  return Sim3(ScSO3(1.,se3.so3()),se3.translation());
}

SE3 Sim3
::to_SE3(const Sim3 & sim3)
{
  return sim3.to_SE3();
}

SE3 Sim3
::to_SE3() const
{
  return to_SE3(*this);
}

Sim3 Sim3
::inverse() const
{
  const ScSO3 sRinv = scso3_.inverse();
  return Sim3(sRinv, -(sRinv*translation_));
}

Sim3& Sim3
::operator*=(const Sim3& sim3)
{
  translation_ += (scso3_ * sim3.translation());
  scso3_ *= sim3.scso3();
  return *this;
}

Sim3 Sim3
::operator*(const Sim3& sim3) const
{
  return Sim3(scso3_*sim3.scso3(),
              (scso3_*sim3.translation()) + translation_);
}

Matrix4d Sim3
::matrix() const
{
  Matrix4d homogenious_matrix;
  homogenious_matrix.setIdentity();
  homogenious_matrix.block(0,0,3,3) = scale()*rotation_matrix();
  homogenious_matrix.col(3).head(3) = translation();
  homogenious_matrix(3,3) = 1;
  return homogenious_matrix;
}

Vector3d Sim3
::operator*(const Vector3d & xyz) const
{
  return (scso3_*xyz) + translation_;
}

Matrix7d Sim3::
Adj() const
{
  Matrix3d R = scso3_.rotationMatrix();
  Matrix7d res;
  res.setZero();

  res.block(0,0,3,3) = scale()*R;
  res.block(0,3,3,3) = SO3::hat(translation_)*R;
  res.block(0,6,3,1) = -translation_;
  res.block(3,3,3,3) = R;
  res(6,6) = 1;

  return res;
}


Matrix3d Sim3
::calcW(double theta,
        double sigma,
        double scale,
        const Matrix3d & Omega)
{
  Matrix3d Omega2 = Omega*Omega;

  static const Matrix3d I = Matrix3d::Identity();

  double A,B,C;
  if (fabs(sigma)<SMALL_EPS)
  {
    C = 1;
    if (theta<SMALL_EPS)
    {
      A = 1./2.;
      B = 1./6.;
      //ToDO: Use more accurate expansion
    }
    else
    {
      double theta_sq = theta*theta;
      A = (1-cos(theta))/theta_sq;
      B = (theta-sin(theta))/(theta_sq*theta);
      //ToDO: Use more accurate expansion
    }
  }
  else
  {
    C=(scale-1)/sigma;
    if (theta<SMALL_EPS)
    {
      double sigma_sq = sigma*sigma;
      A = ((sigma-1)*scale+1)/sigma_sq;
      B = ((0.5*sigma*sigma-sigma+1)*scale)/(sigma_sq*sigma);
     //ToDO: Use more accurate expansion
    }
    else
    {
      double theta_sq = theta*theta;

      double a = scale*sin(theta);
      double b = scale*cos(theta);
      double c = theta_sq+sigma*sigma;
      A = (a*sigma+ (1-b)*theta)/(theta*c);
      B = (C-((b-1)*sigma+a*theta)/(c))*1./(theta_sq);

    }
  }
  return A*Omega + B*Omega2 + C*I;
}

Sim3 Sim3
::exp(const Matrix<double,7,1>& vect)
{
  Vector3d upsilon = vect.segment(0,3);
  Vector3d omega = vect.segment(3,3);
  double sigma = vect[6];

  double theta;
  ScSO3 scso3 = ScSO3::expAndTheta(vect.tail<4>(), &theta);

  Matrix3d Omega = SO3::hat(omega);
  Matrix3d W = calcW(theta, sigma, scso3.scale(), Omega);
  Vector3d t = W*upsilon;
  return Sim3(scso3, t);
}

Matrix<double,7,1> Sim3
::log(const Sim3& sim3)
{
  Vector7d res;
  double scale = sim3.scale();

  Vector3d t = sim3.translation_;

  double theta;

  Vector4d omega_sigma = ScSO3::logAndTheta(sim3.scso3_, &theta);
  Vector3d omega = omega_sigma.head<3>();
  double sigma = omega_sigma[3];
  Matrix3d Omega = SO3::hat(omega);

  Matrix3d W = calcW(theta, sigma, scale, Omega);

  //Vector3d upsilon = W.jacobiSvd(ComputeFullU | ComputeFullV).solve(t);
  Vector3d upsilon = W.partialPivLu().solve(t);

  res.segment(0,3) = upsilon;
  res.segment(3,3) = omega;
  res[6] = sigma;
  return res;
}

Matrix4d Sim3::
hat(const Vector7d & v)
{
  Matrix4d Omega;
  Omega.topLeftCorner<3,3>() = ScSO3::hat(v.tail<4>());
  Omega.col(3).head<3>() = v.head<3>();
  Omega.row(3) = Vector4d(0., 0., 0., 0.);
  return Omega;
}

Vector7d Sim3::
vee(const Matrix4d & Omega)
{
  Vector7d upsilon_omega_sigma;
  upsilon_omega_sigma.head<3>() = Omega.col(3).head<3>();
  upsilon_omega_sigma.tail<4>() = ScSO3::vee(Omega.topLeftCorner<3,3>());
  return upsilon_omega_sigma;
}

Vector7d Sim3
::lieBracket(const Vector7d & v1, const Vector7d & v2)
{
  Vector3d upsilon1 = v1.head<3>();
  Vector3d upsilon2 = v2.head<3>();
  Vector3d omega1 = v1.segment<3>(3);
  Vector3d omega2 = v2.segment<3>(3);
  double sigma1 = v1[6];
  double sigma2 = v2[6];

  Vector7d res;
  res.head<3>() =
      SO3::hat(omega1)*upsilon2 + SO3::hat(upsilon1)*omega2
      + sigma1*upsilon2 - sigma2*upsilon1;
  res.segment<3>(3) = omega1.cross(omega2);
  res[6] = 0.;

  return res;
}

Matrix7d Sim3
::d_lieBracketab_by_d_a(const Vector7d & b)
{
  Matrix7d res;
  res.setZero();

  Vector3d upsilon2 = b.head<3>();
  Vector3d omega2 = b.segment<3>(3);
  double sigma2 = b[6];

  res.topLeftCorner<3,3>() = -SO3::hat(omega2)-sigma2*Matrix3d::Identity();
  res.block<3,3>(0,3) = -SO3::hat(upsilon2);
  res.topRightCorner<3,1>() = upsilon2;

  res.block<3,3>(3,3) = -SO3::hat(omega2);
  return res;
}

SO3 Sim3
::so3() const
{
  return SO3(quaternion());
}

}


