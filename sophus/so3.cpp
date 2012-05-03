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
#include "so3.h"

//ToDo: Think completely through when to normalize Quaternion

namespace Sophus
{

SO3::SO3()
{
  unit_quaternion_.setIdentity();
}

SO3
::SO3(const SO3 & other) : unit_quaternion_(other.unit_quaternion_) {}

SO3
::SO3(const Matrix3d & R) : unit_quaternion_(R) {}

SO3
::SO3(const Quaterniond & quat) : unit_quaternion_(quat)
{
  unit_quaternion_.normalize();
}

SO3
::SO3(double rot_x, double rot_y, double rot_z)
{
  unit_quaternion_
      = (SO3::exp(Vector3d(rot_x, 0.f, 0.f))
         *SO3::exp(Vector3d(0.f, rot_y, 0.f))
         *SO3::exp(Vector3d(0.f, 0.f, rot_z))).unit_quaternion_;
}

void SO3
::operator=(const SO3 & other)
{
  this->unit_quaternion_ = other.unit_quaternion_;
}

SO3 SO3
::operator*(const SO3& other) const
{
  SO3 result(*this);
  result.unit_quaternion_ *= other.unit_quaternion_;
  result.unit_quaternion_.normalize();
  return result;
}

void SO3
::operator*=(const SO3& other)
{
  unit_quaternion_ *= other.unit_quaternion_;
  unit_quaternion_.normalize();
}

Vector3d SO3
::operator*(const Vector3d & xyz) const
{
  return unit_quaternion_._transformVector(xyz);
}

SO3 SO3
::inverse() const
{
  return SO3(unit_quaternion_.conjugate());
}

Matrix3d SO3
::matrix() const
{
  return unit_quaternion_.toRotationMatrix();
}

Matrix3d SO3
::Adj() const
{
  return matrix();
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
  double theta;
  return logAndTheta(other, &theta);
}

Vector3d SO3
::logAndTheta(const SO3 & other, double * theta)
{
  double q_real = other.unit_quaternion_.w();


  if (q_real>1.-SMALL_EPS)
  {
    *theta = 2.*acos(std::min(q_real, 1.0));
    return (2.-2./3.*(q_real-1.)+4./15.*(q_real-1.)*(q_real-1.))
        *Vector3d(other.unit_quaternion_.x(),
                  other.unit_quaternion_.y(),
                  other.unit_quaternion_.z());
  }

  *theta = 2.*acos(q_real);
  double theta_by_sin_half_theta = (*theta)/sqrt(1. - q_real*q_real);

  return Vector3d(theta_by_sin_half_theta*other.unit_quaternion_.x(),
                  theta_by_sin_half_theta*other.unit_quaternion_.y(),
                  theta_by_sin_half_theta*other.unit_quaternion_.z());
}

SO3 SO3
::exp(const Vector3d & omega)
{
  double theta;
  return expAndTheta(omega, &theta);
}

SO3 SO3
::expAndTheta(const Vector3d & omega, double * theta)
{
  *theta = omega.norm();
  double half_theta = 0.5*(*theta);

  double imag_factor;
  double real_factor = cos(half_theta);
  if((*theta)<SMALL_EPS)
  {
    double theta_sq = (*theta)*(*theta);
    double theta_po4 = theta_sq*theta_sq;
    imag_factor = 0.5-0.0208333*theta_sq+0.000260417*theta_po4;
  }
  else
  {
    double sin_half_theta = sin(half_theta);
    imag_factor = sin_half_theta/(*theta);
  }

  return SO3(Quaterniond(real_factor,
                         imag_factor*omega.x(),
                         imag_factor*omega.y(),
                         imag_factor*omega.z()));
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
  assert(fabs(Omega(2,1)+Omega(1,2))<SMALL_EPS);
  assert(fabs(Omega(0,2)+Omega(2,0))<SMALL_EPS);
  assert(fabs(Omega(1,0)+Omega(0,1))<SMALL_EPS);
  return Vector3d(Omega(2,1), Omega(0,2), Omega(1,0));
}

Vector3d SO3
::lieBracket(const Vector3d & omega1, const Vector3d & omega2)
{
  return omega1.cross(omega2);
}

Matrix3d SO3
::d_lieBracketab_by_d_a(const Vector3d & b)
{
  return -hat(b);
}

void SO3::
setQuaternion(const Quaterniond& quaternion)
{
  assert(quaternion.norm()!=0);
  unit_quaternion_ = quaternion;
  unit_quaternion_.normalize();
}


}
