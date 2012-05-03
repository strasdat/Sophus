// This file is part of Sophus.
//
// Copyright 2012 Hauke Strasdat (Imperial College London)
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
#include "scso3.h"

namespace Sophus
{

ScSO3::ScSO3()
{
  quaternion_.setIdentity();
}

ScSO3
::ScSO3(const ScSO3 & other) : quaternion_(other.quaternion_) {}

ScSO3
::ScSO3(const Matrix3d & scale_times_R)
{
  Matrix3d squaredScaleMat = scale_times_R*scale_times_R.transpose();
  double squaredScale
      =( 1./3.)*(squaredScaleMat(0,0)
                 +squaredScaleMat(1,1)
                 +squaredScaleMat(2,2));
  assert(squaredScale>0);
  double scale = sqrt(squaredScale);

  quaternion_ = scale_times_R/scale;
  quaternion_.coeffs() *= scale;
}


ScSO3
::ScSO3(double scale,
        const Matrix3d & R)
{
  assert(scale>0);
  quaternion_ = R;
  quaternion_.normalize();
  quaternion_.coeffs() *= scale;
}

ScSO3
::ScSO3(double scale, const SO3 & so3)
{
  assert(scale>0);
  quaternion_ = so3.unit_quaternion();
  quaternion_.coeffs() *= scale;
}

ScSO3
::ScSO3(const Quaterniond & quat) : quaternion_(quat) {}

void ScSO3
::operator=(const ScSO3 & other)
{
  this->quaternion_ = other.quaternion_;
}

ScSO3 ScSO3
::operator*(const ScSO3& other) const
{
  ScSO3 result(*this);
  result.quaternion_ *= other.quaternion_;
  return result;
}

void ScSO3
::operator*=(const ScSO3& other)
{
  quaternion_ *= other.quaternion_;
}

Vector3d ScSO3
::operator*(const Vector3d & xyz) const
{
  //ToDO: implement this directly!
  double scale = quaternion().norm();
  Quaterniond norm_quad = quaternion_;
  norm_quad.coeffs() /= scale;
  return scale*norm_quad._transformVector(xyz);
}

ScSO3 ScSO3
::inverse() const
{
  assert(quaternion_.squaredNorm()>0.);
  return ScSO3(quaternion().inverse());
}

Matrix3d ScSO3
::matrix() const
{
  //ToDO: implement this directly!
  double scale = quaternion().norm();
  Quaterniond norm_quad = quaternion_;
  norm_quad.coeffs() /= scale;
  return scale*norm_quad.toRotationMatrix();
}

Matrix3d ScSO3
::rotationMatrix() const
{
  double scale = quaternion().norm();
  Quaterniond norm_quad = quaternion_;
  norm_quad.coeffs() /= scale;
  return norm_quad.toRotationMatrix();
}

double ScSO3
::scale() const
{
  return quaternion_.norm();
}

void ScSO3
::setRotationMatrix(const Matrix3d & R)
{
  double scale = quaternion_.norm();
  quaternion_ = R;
  quaternion_.normalize(); //is this step necesarry??
  quaternion_.coeffs() *= scale;
}

void ScSO3
::setScale(double scale)
{
  quaternion_.normalize();
  quaternion_.coeffs() *= scale;
}

Matrix4d ScSO3
::Adj() const
{
  Matrix4d res;
  res.setIdentity();
  res.topLeftCorner<3,3>() = rotationMatrix();
  return res;
}

Matrix3d ScSO3
::generator(int i)
{
  assert(i>=0 && i<4);
  Vector4d e;
  e.setZero();
  e[i] = 1.f;
  return hat(e);
}

Vector4d ScSO3
::log() const
{
  return ScSO3::log(*this);
}

Vector4d ScSO3
::log(const ScSO3 & other)
{
  double theta;
  return logAndTheta(other, &theta);
}

Vector4d ScSO3
::logAndTheta(const ScSO3 & other, double * theta)
{
  double scale = other.quaternion().norm();
  Vector4d omega_sigma;
  omega_sigma[3] = std::log(scale);
  omega_sigma.head<3>() = SO3::logAndTheta(SO3(other.quaternion()), theta);
  return omega_sigma;
}

ScSO3 ScSO3
::exp(const Vector4d & omega_sigma)
{
  double theta;
  return expAndTheta(omega_sigma, &theta);
}

ScSO3 ScSO3
::expAndTheta(const Vector4d & omega_sigma, double * theta)
{
  const Vector3d & omega = omega_sigma.head<3>();
  double sigma = omega_sigma[3];
  double scale = std::exp(sigma);

  Quaterniond quat = SO3::expAndTheta(omega, theta).unit_quaternion();
  quat.coeffs() *= scale;

  return ScSO3(quat);
}

Matrix3d ScSO3
::hat(const Vector4d & v)
{
  Matrix3d Omega;
  Omega <<  v(3), -v(2),  v(1)
      ,  v(2),     v(3), -v(0)
      , -v(1),  v(0),     v(3);
  return Omega;
}

Vector4d ScSO3
::vee(const Matrix3d & Omega)
{
  assert(fabs(Omega(2,1)+Omega(1,2))<SMALL_EPS);
  assert(fabs(Omega(0,2)+Omega(2,0))<SMALL_EPS);
  assert(fabs(Omega(1,0)+Omega(0,1))<SMALL_EPS);
  assert(fabs(Omega(0,0)-Omega(1,1))<SMALL_EPS);
  assert(fabs(Omega(0,0)-Omega(2,2))<SMALL_EPS);
  return Vector4d(Omega(2,1), Omega(0,2), Omega(1,0), Omega(0,0));
}

Vector4d ScSO3
::lieBracket(const Vector4d & omega_sigma1, const Vector4d & omega_sigma2)
{
  Vector3d omega1 = omega_sigma1.head<3>();
  Vector3d omega2 = omega_sigma2.head<3>();
  Vector4d res;
  res.head<3>() = omega1.cross(omega2);
  res[3] = 0.;
  return res;
}

Matrix4d ScSO3
::d_lieBracketab_by_d_a(const Vector4d & b)
{
  Matrix4d res;
  res.setZero();
  res.topLeftCorner<3,3>() = -SO3::hat(b.head<3>());
  return res;
}


}
