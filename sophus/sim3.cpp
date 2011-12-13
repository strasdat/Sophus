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

namespace Sophus
{
Sim3
::Sim3()
{
  translation_.setZero();
  scale_ = 1.;
}

Sim3
::Sim3(const SO3& so3, const Vector3d & t, double s)
  :so3_(so3),translation_(t),scale_(s)
{
}

Sim3
::Sim3(const Matrix3d& R, const Vector3d & t, double s)
  :so3_(R),translation_(t),scale_(s)
{
}

Sim3
::Sim3(const Quaterniond& q, const Vector3d & t,double s)
  :so3_(q),translation_(t),scale_(s)
{
}

Sim3
::Sim3(const Sim3 & sim3)
  : so3_(sim3.so3_),translation_(sim3.translation_),scale_(sim3.scale_)
{
}

Sim3 Sim3
::from_SE3(const SE3 & se3)
{
  return Sim3(se3.so3(),se3.translation(),1);
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
  const SO3 Rinv = so3().inverse();
  return Sim3(Rinv, -(1./scale_)*(Rinv*translation_), 1./scale_);
}

Sim3& Sim3
::operator*=(const Sim3& sim3)
{
  translation_ += scale_*(so3_ * sim3.translation());
  so3_ *= sim3.so3();
  scale_ *= sim3.scale();
  return *this;
}

Sim3 Sim3
::operator*(const Sim3& sim3) const
{
  return Sim3(so3_*sim3.so3(),
              scale_*(so3_*sim3.translation()) + translation_,
              scale_*sim3.scale());
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
  return scale_*(so3_*xyz) + translation_;
}

Matrix7d Sim3::
Adj() const
{
  Matrix3d R = so3_.matrix();
  Matrix7d res;
  res.setZero();

  res.block(0,0,3,3) = scale_*R;
  res.block(0,3,3,3) = SO3::hat(translation_)*R;
  res.block(0,6,3,1) = -translation_;
  res.block(3,3,3,3) = R;
  res(6,6) = 1;

  return res;
}

Sim3 Sim3
::exp(const Matrix<double,7,1>& vect)
{
  Vector3d upsilon = vect.segment(0,3);
  Vector3d omega = vect.segment(3,3);
  double sigma = vect[6];

  double theta = omega.norm();
  Matrix3d Omega = SO3::hat(omega);
  Matrix3d Omega2 = Omega*Omega;
  Matrix3d R;

  Matrix3d I;
  I.setIdentity();
  double s = std::exp(sigma);

  double A,B,C;
  if (fabs(sigma)<SMALL_EPS)
  {
    C = 1;
    if (theta<SMALL_EPS)
    {
      A = 1./2.;
      B = 1./6.;
      R = (I + Omega + Omega*Omega);
    }
    else
    {
      double theta_sq = theta*theta;
      A = (1-cos(theta))/theta_sq;
      B = (theta-sin(theta))/(theta_sq*theta);
      R = I + sin(theta)/theta *Omega + (1-cos(theta))/(theta_sq)*Omega2;
    }
  }
  else
  {
    C=(s-1)/sigma;
    if (theta<SMALL_EPS)
    {
      double theta_sq = theta*theta;
      A = ((sigma-1)*s+1)/theta_sq;
      B = ((0.5*sigma*sigma-sigma+1)*s)/(theta_sq*theta);
      R = (I + Omega + Omega2);
    }
    else
    {
      double theta_sq = theta*theta;
      R = I + sin(theta)/theta *Omega + (1-cos(theta))/(theta_sq)*Omega2;
      double a = s*sin(theta);
      double b = s*cos(theta);
      double c = theta_sq+sigma*sigma;
      A = (a*sigma+ (1-b)*theta)/(theta*c);
      B = (C-((b-1)*sigma+a*theta)/(c))*1./(theta_sq);

    }
  }

  Matrix3d W = A*Omega + B*Omega2 + C*I;
  Vector3d t = W*upsilon;
  return Sim3(R, t, s);
}

Matrix<double,7,1> Sim3
::log(const Sim3& sim3)
{
  Vector7d res;
  double s = sim3.scale_;
  double sigma = std::log(s);

  Vector3d t = sim3.translation_;
  const Matrix3d & R = sim3.rotation_matrix();
  double d =  0.5*(R(0,0)+R(1,1)+R(2,2)-1);

  Vector3d omega;
  Vector3d upsilon;
  Matrix3d Omega;

  Matrix3d I;
  I.setIdentity();

  double A,B,C;
  if (fabs(sigma)<SMALL_EPS)
  {
    C = 1;
    if (d > 1.-SMALL_EPS)
    {
      omega=0.5*SO3::deltaR(R);
      Omega = SO3::hat(omega);
      A = 1./2.;
      B = 1./6.;
    }
    else
    {
      double theta = acos(d);
      double theta_sq = theta*theta;
      omega = theta/(2*sqrt(1-d*d))*SO3::deltaR(R);
      Omega = SO3::hat(omega);
      A = (1-cos(theta))/theta_sq;
      B = (theta-sin(theta))/(theta_sq*theta);
    }
  }
  else
  {
    C=(s-1)/sigma;
    if (d > 1.-SMALL_EPS)
    {
      omega=0.5*SO3::deltaR(R);
      Omega = SO3::hat(omega);
      double sigma_sq = sigma*sigma;
      A = ((sigma-1)*s+1)/sigma_sq;
      B = ((0.5*sigma*sigma-sigma+1)*s)/(sigma_sq*sigma);
    }
    else
    {
      double theta = acos(d);
      double theta_sq = theta*theta;
      omega = theta/(2*sqrt(1-d*d))*SO3::deltaR(R);
      Omega = SO3::hat(omega);
      double a = s*sin(theta);
      double b = s*cos(theta);
      double c = theta_sq+sigma*sigma;
      A = (a*sigma+ (1-b)*theta)/(theta*c);
      B = (C-((b-1)*sigma+a*theta)/(c))*1./(theta_sq);
    }
  }
  Matrix3d W = A*Omega + B*Omega*Omega + C*I;
  upsilon = W.lu().solve(t);
  res.segment(0,3) = upsilon;
  res.segment(3,3) = omega;
  res[6] = sigma;
  return res;
}

Matrix4d Sim3::
hat(const Vector7d & v)
{
  Matrix4d Omega;
  Omega.topLeftCorner<3,3>() = SO3::hat(v.tail<3>());
  Omega(0,0) = v[6];
  Omega(1,1) = v[6];
  Omega(2,2) = v[6];
  Omega.col(3).head<3>() = v.head<3>();
  Omega.row(3) = Vector4d(0., 0., 0., 1.);
  return Omega;
}

Vector7d Sim3::
vee(const Matrix4d & Omega)
{
  assert(fabs(Omega(0,0)-Omega(1,1))<SMALL_EPS);
  assert(fabs(Omega(1,1)-Omega(2,2))<SMALL_EPS);
  Vector7d upsilon_omega_sigma;
  upsilon_omega_sigma.head<3>() = Omega.col(3).head<3>();
  upsilon_omega_sigma.segment<3>(3) = SO3::vee(Omega.topLeftCorner<3,3>());
  upsilon_omega_sigma[6] = Omega(0,0);
  return upsilon_omega_sigma;
}

Vector7d Sim3::
lieBracket(const Vector7d & omega1, const Vector7d & omega2)
{
  return vee(hat(omega1)*hat(omega2)-hat(omega2)*hat(omega1));
}

}


