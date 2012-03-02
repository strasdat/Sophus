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
  Omega.topLeftCorner<3,3>() = SO3::hat(v.tail<3>());
  Omega.col(3).head<3>() = v.head<3>();
  Omega.row(3) = Vector4d(0., 0., 0., 1.);
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
  return vee(SE3::hat(v1)*SE3::hat(v2) - SE3::hat(v2)*SE3::hat(v1));
}

SE3 SE3
::exp(const Vector6d & update)
{
  Quaterniond q;
  Vector3d upsilon = update.head<3>();
  Vector3d omega = update.tail<3>();

  double theta = omega.norm();
  Matrix3d Omega = SO3::hat(omega);
  Matrix3d Omega_sq = Omega*Omega;

  Matrix3d V;
  double half_theta = 0.5*theta;

  double imag_factor;
  double real_factor = cos(half_theta);
  if(theta<SMALL_EPS)
  {
    double theta_sq = theta*theta;
    double theta_po4 = theta_sq*theta_sq;
    imag_factor = 0.5-0.0208333*theta_sq+0.000260417*theta_po4;
    q.w() = real_factor;
    q.x() = imag_factor*omega.x();
    q.y() = imag_factor*omega.y();
    q.z() = imag_factor*omega.z();
    V = q.matrix();
  }
  else
  {
    double theta_sq = theta*theta;
    double sin_half_theta = sin(half_theta);
    imag_factor = sin_half_theta/theta;
    q.w() = real_factor;
    q.x() = imag_factor*omega.x();
    q.y() = imag_factor*omega.y();
    q.z() = imag_factor*omega.z();
    V = (Matrix3d::Identity()
         + (1-cos(theta))/(theta_sq)*Omega
         + (theta-sin(theta))/(theta_sq*theta)*Omega_sq);
  }
  return SE3(q,V*upsilon);
}

//Vector6d SE3
//::log(const SE3 & se3)
//{
//  Vector6d upsilon_omega;
//  double q_real = se3.so3_.quaternion().w();
//  double theta ;
//  Matrix3d _R = se3.so3_.matrix();
//  Vector3d dR = SO3::deltaR(_R);
//  Vector3d omega;

//  if (q_real>1.-SMALL_EPS)
//  {
//    upsilon_omega.tail<3>() =  (2.-2./3.*(q_real-1.)+4./15.*(q_real-1.)*(q_real-1.))
//        *Vector3d(se3.so3_.quaternion().x(),se3.so3_.quaternion().y(),se3.so3_.quaternion().z());
//    theta = upsilon_omega.tail<3>().norm();
//    omega = 0.5*dR;
//    Matrix3d Omega = SO3::hat(omega);
//    Matrix3d V_inv = Matrix3d::Identity()- 0.5*Omega + (1./12.)*(Omega*Omega);
//    upsilon_omega.head<3>() = V_inv*se3.translation_;
//  }
//  else
//  {

//    theta = 2.*acos(q_real);
//    double theta_by_sin_half_theta = theta/sqrt(1. - q_real*q_real);

//    upsilon_omega.tail<3>() = Vector3d(theta_by_sin_half_theta*se3.so3_.quaternion().x(),
//                                       theta_by_sin_half_theta*se3.so3_.quaternion().y(),
//                                       theta_by_sin_half_theta*se3.so3_.quaternion().z());

//    Matrix3d Omega = SO3::hat(upsilon_omega.tail<3>());
//    Matrix3d V_inv = ( Matrix3d::Identity() - 0.5*Omega
//              + ( 1-theta/(2*tan(theta/2)))/(theta*theta)*(Omega*Omega) );
//    upsilon_omega.head<3>() = V_inv*se3.translation_;

//  }

//  return upsilon_omega;

//}

//SE3 SE3
//::exp(const Vector6d & update)
//{
//  Vector3d upsilon = update.head<3>();
//  Vector3d omega = update.tail<3>();

//  double theta = omega.norm();
//  Matrix3d Omega = SO3::hat(omega);

//  Matrix3d R;
//  Matrix3d V;
//  if (theta<SMALL_EPS)
//  {
//    R = (Matrix3d::Identity() + Omega + Omega*Omega);
//    V = R;
//  }
//  else
//  {
//    Matrix3d Omega2 = Omega*Omega;

//    R = (Matrix3d::Identity()
//         + sin(theta)/theta *Omega
//         + (1-cos(theta))/(theta*theta)*Omega2);

//    V = (Matrix3d::Identity()
//         + (1-cos(theta))/(theta*theta)*Omega
//         + (theta-sin(theta))/(pow(theta,3))*Omega2);
//  }
//  return SE3(Quaterniond(R),V*upsilon);
//}

Vector6d SE3
::log(const SE3 & se3)
{
  Vector6d res;
  Matrix3d _R = se3.so3_.matrix();
  double d = 0.5*(_R(0,0)+_R(1,1)+_R(2,2)-1);
  Vector3d omega;
  Vector3d upsilon;
  Vector3d dR = SO3::deltaR(_R);
  Matrix3d V_inv;
  if (d>1.-SMALL_EPS)
  {
    omega = 0.5*dR;
    Matrix3d Omega = SO3::hat(omega);
    V_inv = Matrix3d::Identity()- 0.5*Omega + (1./12.)*(Omega*Omega);
  }
  else
  {
    double theta = acos(d);
    omega = theta/(2*sqrt(1-d*d))*dR;
    Matrix3d Omega = SO3::hat(omega);
    V_inv = ( Matrix3d::Identity() - 0.5*Omega
              + ( 1-theta/(2*tan(theta/2)))/(theta*theta)*(Omega*Omega) );
  }
  upsilon = V_inv*se3.translation_;

  res.head<3>() = upsilon;
  res.tail<3>() = omega;
  return res;
}

}

