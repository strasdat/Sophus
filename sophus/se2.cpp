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

#include <iostream>
#include "se2.h"
#include "so3.h"


namespace Sophus
{
SE2
::SE2()
{
  translation_.setZero();
}

SE2
::SE2(const SO2 & so2, const Vector2d & translation)
  : so2_(so2), translation_(translation) {}

SE2
::SE2(const Matrix2d & rotation_matrix, const Vector2d & translation)
  : so2_(rotation_matrix), translation_(translation){}

SE2
::SE2(double theta, const Vector2d & translation)
  : so2_(theta), translation_(translation) {}

SE2
::SE2(const SE2 & se2) : so2_(se2.so2_),translation_(se2.translation_){}


SE2 & SE2
::operator = (const SE2 & other)
{
  so2_ = other.so2_;
  translation_ = other.translation_;
  return *this;
}

SE2 SE2
::operator*(const SE2 & other) const
{
  SE2 result(*this);
  result.translation_ += so2_*(other.translation_);
  result.so2_*=other.so2_;
  return result;
}

SE2& SE2
::operator *= (const SE2 & other)
{
  translation_+= so2_*(other.translation_);
  so2_*=other.so2_;
  return *this;
}

SE2 SE2
::inverse() const
{
  SE2 ret;
  ret.so2_= so2_.inverse();
  ret.translation_ = ret.so2_*(translation_*-1.);
  return ret;
}

Vector3d SE2
::log() const
{
  return log(*this);
}

Vector2d SE2
::operator *(const Vector2d & xyz) const
{
  return so2_*xyz + translation_;
}

Matrix3d SE2
::matrix() const
{
  Matrix<double,3,3> homogenious_matrix;
  homogenious_matrix.setIdentity();
  homogenious_matrix.block(0,0,2,2) = rotation_matrix();
  homogenious_matrix.col(2).head(2) = translation_;
  return homogenious_matrix;
}


Matrix<double, 3, 3> SE2
::Adj() const
{
  const Matrix2d & R = so2_.matrix();
  Matrix3d res;
  res.setIdentity();
  res.topLeftCorner<2,2>() = R;
  res(0,2) = translation_[1];
  res(1,2) = -translation_[0];
  return res;
}

Matrix3d SE2
::hat(const Vector3d & v)
{
  Matrix3d Omega;
  Omega.setZero();
  Omega.topLeftCorner<2,2>() = SO2::hat(v[2]);
  Omega.col(2).head<2>() = v.head<2>();
  return Omega;
}

Vector3d SE2
::vee(const Matrix3d & Omega)
{
  Vector3d upsilon_omega;
  upsilon_omega.head<2>() = Omega.col(2).head<2>();
  upsilon_omega[2] = SO2::vee(Omega.topLeftCorner<2,2>());
  return upsilon_omega;
}

Vector3d SE2
::lieBracket(const Vector3d & v1, const Vector3d & v2)
{
  Vector2d upsilon1 = v1.head<2>();
  Vector2d upsilon2 = v2.head<2>();
  double theta1 = v1[2];
  double theta2 = v2[2];

  return Vector3d(-theta1*upsilon2[1] + theta2*upsilon1[1],
                  theta1*upsilon2[0] - theta2*upsilon1[0],
                  0.);
}

Matrix3d SE2
::d_lieBracketab_by_d_a(const Vector3d & v2)
{
  Vector2d upsilon2 = v2.head<2>();
  double theta2 = v2[2];

  Matrix3d res;
  res <<      0., theta2, -upsilon2[1]
      ,  -theta2,     0.,  upsilon2[0]
      ,       0.,     0.,           0.;
  return res;
}

SE2 SE2
::exp(const Vector3d & update)
{
  Vector2d upsilon = update.head<2>();
  double theta = update[2];

  SO2 so2 = SO2::exp(theta);


  double sin_theta_by_theta;
  double one_minus_cos_theta_by_theta;
  if(abs(theta)<SMALL_EPS)
  {
    double theta_sq = theta*theta;

    sin_theta_by_theta
        = 1.-(1./6.)*theta_sq;
    one_minus_cos_theta_by_theta
        = 0.5*theta
        - (1./24.)*theta*theta_sq;
  }
  else
  {
    sin_theta_by_theta
        = so2.unit_complex().imag()/theta;
    one_minus_cos_theta_by_theta
        = (1.-so2.unit_complex().real())/theta;
  }
  Matrix2d V;
  V(0,0) = sin_theta_by_theta; V(0,1) = -one_minus_cos_theta_by_theta;
  V(1,0) = one_minus_cos_theta_by_theta; V(1,1) = sin_theta_by_theta;
  return SE2(so2,V*upsilon);
}

Vector3d SE2
::log(const SE2 & se2)
{
  Vector3d upsilon_theta;
  const SO2 & so2 = se2.so2();
  double theta = SO2::log(so2);
  upsilon_theta[2] = theta;

  double halftheta = 0.5*theta;

  double halftheta_by_tan_of_halftheta;
  if (abs(theta)<SMALL_EPS)
  {
    halftheta_by_tan_of_halftheta = 1. - (1./12)*theta*theta;
  }
  else
  {
    const SO2::Complexd & z = so2.unit_complex();
    halftheta_by_tan_of_halftheta = -(halftheta*z.imag())/(z.real()-1.);
  }
  Matrix2d V_inv;
  V_inv(0,0) = halftheta_by_tan_of_halftheta; V_inv(1,0) = -halftheta;
  V_inv(0,1) = halftheta; V_inv(1,1) = halftheta_by_tan_of_halftheta;
  upsilon_theta.head<2>() = V_inv*se2.translation_;
  return upsilon_theta;
}

}

