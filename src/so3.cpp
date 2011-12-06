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

#include "so3.h"

namespace Sophus
{

SO3::SO3()
{
  quaternion_.setIdentity();
}

SO3
::SO3(const SO3 & other) : quaternion_(other.quaternion_) {}

SO3
::SO3(const Matrix3d & R) : quaternion_(R) {}

SO3
::SO3(const Quaterniond & quat) : quaternion_(quat) {}

SO3
::SO3(double rot_x, double rot_y, double rot_z)
{
  quaternion_
      = (SO3::exp(Vector3d(rot_x, 0.f, 0.f))
         *SO3::exp(Vector3d(0.f, rot_y, 0.f))
         *SO3::exp(Vector3d(0.f, 0.f, rot_z))).quaternion_;
}

void SO3
::operator=(const SO3 & other)
{
  this->quaternion_ = other.quaternion_;
}

SO3 SO3
::operator*(const SO3& other) const
{
  SO3 result(*this);
  result.quaternion_ *= other.quaternion_;
  result.quaternion_.normalize();
  return result;
}

void SO3
::operator*=(const SO3& other)
{
  quaternion_ *= other.quaternion_;
  quaternion_.normalize();
}

Vector3d SO3
::operator*(const Vector3d & xyz) const
{
  return quaternion_._transformVector(xyz);
}

SO3 SO3
::inverse() const
{
  return SO3(quaternion_.conjugate());
}

Matrix3d SO3
::matrix() const
{
  return quaternion_.toRotationMatrix();
}

Matrix3d SO3
::Adj() const
{
  return quaternion_.toRotationMatrix();
}

Matrix3d SO3
::generator(int i)
{
  assert(i>=0 && i<3);
  Vector3d e;
  e.setZero();
  e[i] = 1.f;
  return hat(e);
}

Vector3d SO3
::log() const
{
  return SO3::log(*this);
}

Vector3d SO3
::log(const SO3 & other)
{
  double q_real = other.quaternion_.w();
  if (q_real>1.-SMALL_EPS)
  {
    return Vector3d(0.,0.,0.);
  }

  double theta = 2.*acos(q_real);
  double theta_by_sin_half_theta = theta/sqrt(1. - q_real*q_real);

  return Vector3d(theta_by_sin_half_theta*other.quaternion_.x(),
                  theta_by_sin_half_theta*other.quaternion_.y(),
                  theta_by_sin_half_theta*other.quaternion_.z());
}

SO3 SO3
::exp(const Vector3d & omega)
{
  double theta = omega.norm();

  if(theta<SMALL_EPS)
  {
    return SO3(Quaterniond::Identity());
  }

  double half_theta = 0.5*theta;
  double sin_half_theta = sin(half_theta);
  double sin_half_theta_by_theta = sin_half_theta/theta;
  return SO3(Quaterniond(cos(half_theta),
                         sin_half_theta_by_theta*omega.x(),
                         sin_half_theta_by_theta*omega.y(),
                         sin_half_theta_by_theta*omega.z()));
}

Matrix3d SO3
::hat(const Vector3d & v)
{
  Matrix3d Omega;
  Omega <<  0, -v(2),  v(1)
      ,  v(2),     0, -v(0)
      , -v(1),  v(0),     0;
  return Omega;
}

Vector3d SO3
::vee(const Matrix3d & Omega)
{
  assert(fabs(Omega(2,1)-Omega(1,2))<SMALL_EPS);
  assert(fabs(Omega(0,2)-Omega(2,0))<SMALL_EPS);
  assert(fabs(Omega(1,0)-Omega(0,1))<SMALL_EPS);
  return Vector3d(Omega(2,1), Omega(0,2), Omega(1,0));
}

Vector3d SO3
::lieBracket(const Vector3d & omega1, const Vector3d & omega2)
{
  return omega1.cross(omega2);
}

Vector3d SO3::
deltaR(const Matrix3d & R)
{
  Vector3d v;
  v(0) = R(2,1)-R(1,2);
  v(1) = R(0,2)-R(2,0);
  v(2) = R(1,0)-R(0,1);
  return v;
}

}
