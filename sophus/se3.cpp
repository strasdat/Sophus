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
#include "se3.h"


namespace Sophus
{
SE3
::SE3()
{
  translation_.setZero();
}

SE3
::SE3(const SO3 & so3, const Vector3d & translation)
  : so3_(so3), translation_(translation) {}

SE3
::SE3(const Matrix3d & rotation_matrix, const Vector3d & translation)
  : so3_(rotation_matrix), translation_(translation){}

SE3
::SE3(const Quaterniond & quaternion, const Vector3d & translation)
  : so3_(quaternion), translation_(translation) {}

SE3
::SE3(const SE3 & se3) : so3_(se3.so3_),translation_(se3.translation_){}


SE3 & SE3
::operator = (const SE3 & other)
{
  so3_ = other.so3_;
  translation_ = other.translation_;
  return *this;
}

SE3 SE3
::operator*(const SE3 & other) const
{
  SE3 result(*this);
  result.translation_ += so3_*(other.translation_);
  result.so3_*=other.so3_;
  return result;
}

SE3& SE3
::operator *= (const SE3 & other)
{
  translation_+= so3_*(other.translation_);
  so3_*=other.so3_;
  return *this;
}

SE3 SE3
::inverse() const
{
  SE3 ret;
  ret.so3_= so3_.inverse();
  ret.translation_= ret.so3_*(translation_*-1.);
  return ret;
}

Vector6d SE3
::log() const
{
  return log(*this);
}

Vector3d SE3
::operator *(const Vector3d & xyz) const
{
  return so3_*xyz + translation_;
}

Matrix4d SE3
::matrix() const
{
  Matrix<double,4,4> homogenious_matrix;
  homogenious_matrix.setIdentity();
  homogenious_matrix.block(0,0,3,3) = rotation_matrix();
  homogenious_matrix.col(3).head(3) = translation_;
  return homogenious_matrix;
}


Matrix<double, 6, 6> SE3
::Adj() const
{
  Matrix3d R = so3_.matrix();
  Matrix<double, 6, 6> res;
  res.block(0,0,3,3) = R;
  res.block(3,3,3,3) = R;
  res.block(0,3,3,3) = SO3::hat(translation_)*R;
  res.block(3,0,3,3) = Matrix3d::Zero(3,3);
  return res;
}

Matrix4d SE3
::hat(const Vector6d & v)
{
  Matrix4d Omega;
  Omega.setZero();
  Omega.topLeftCorner<3,3>() = SO3::hat(v.tail<3>());
  Omega.col(3).head<3>() = v.head<3>();
  return Omega;
}

Vector6d SE3
::vee(const Matrix4d & Omega)
{
  Vector6d upsilon_omega;
  upsilon_omega.head<3>() = Omega.col(3).head<3>();
  upsilon_omega.tail<3>() = SO3::vee(Omega.topLeftCorner<3,3>());
  return upsilon_omega;
}

Vector6d SE3
::lieBracket(const Vector6d & v1, const Vector6d & v2)
{
  Vector3d upsilon1 = v1.head<3>();
  Vector3d upsilon2 = v2.head<3>();
  Vector3d omega1 = v1.tail<3>();
  Vector3d omega2 = v2.tail<3>();

  Vector6d res;
  res.head<3>() = omega1.cross(upsilon2) + upsilon1.cross(omega2);
  res.tail<3>() = omega1.cross(omega2);

  return res;
}

Matrix6d SE3
::d_lieBracketab_by_d_a(const Vector6d & b)
{
  Matrix6d res;
  res.setZero();

  Vector3d upsilon2 = b.head<3>();
  Vector3d omega2 = b.tail<3>();

  res.topLeftCorner<3,3>() = -SO3::hat(omega2);
  res.topRightCorner<3,3>() = -SO3::hat(upsilon2);

  res.bottomRightCorner<3,3>() = -SO3::hat(omega2);
  return res;
}

SE3 SE3
::exp(const Vector6d & update)
{
  Vector3d upsilon = update.head<3>();
  Vector3d omega = update.tail<3>();

  double theta;
  SO3 so3 = SO3::expAndTheta(omega, &theta);

  Matrix3d Omega = SO3::hat(omega);
  Matrix3d Omega_sq = Omega*Omega;
  Matrix3d V;

  if(theta<SMALL_EPS)
  {
    V = so3.matrix();
    //Note: That is an accurate expansion!
  }
  else
  {
    double theta_sq = theta*theta;
    V = (Matrix3d::Identity()
         + (1-cos(theta))/(theta_sq)*Omega
         + (theta-sin(theta))/(theta_sq*theta)*Omega_sq);
  }
  return SE3(so3,V*upsilon);
}

Vector6d SE3
::log(const SE3 & se3)
{
  Vector6d upsilon_omega;
  double theta;
  upsilon_omega.tail<3>() = SO3::logAndTheta(se3.so3_, &theta);

  if (theta<SMALL_EPS)
  {
    Matrix3d Omega = SO3::hat(upsilon_omega.tail<3>());
    Matrix3d V_inv = Matrix3d::Identity()- 0.5*Omega + (1./12.)*(Omega*Omega);

    upsilon_omega.head<3>() = V_inv*se3.translation_;
  }
  else
  {
    Matrix3d Omega = SO3::hat(upsilon_omega.tail<3>());
    Matrix3d V_inv = ( Matrix3d::Identity() - 0.5*Omega
              + ( 1-theta/(2*tan(theta/2)))/(theta*theta)*(Omega*Omega) );
    upsilon_omega.head<3>() = V_inv*se3.translation_;
  }
  return upsilon_omega;
}

void SE3::
setQuaternion(const Quaterniond& quat)
{
  return so3_.setQuaternion(quat);
}

}

