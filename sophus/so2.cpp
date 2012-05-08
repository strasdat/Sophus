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
#include "so2.h"
#include "so3.h"

namespace Sophus
{

SO2::SO2()
{
  unit_complex_.real() = 1.;
  unit_complex_.imag() = 0.;
}

SO2
::SO2(const SO2 & other) : unit_complex_(other.unit_complex_) {}

SO2
::SO2(const Matrix2d & R) : unit_complex_(0.5*(R(0,0)+R(1,1)),
                                          0.5*(R(1,0)-R(0,1))) {}

SO2
::SO2(const Complexd & complex) : unit_complex_(complex)
{
  assert(abs(unit_complex_)!=0);
  normalize();
}

SO2
::SO2(double theta)
{
  unit_complex_ = SO2::exp(theta).unit_complex_;
}

void SO2
::operator=(const SO2 & other)
{
  this->unit_complex_ = other.unit_complex_;
}

SO2 SO2
::operator*(const SO2& other) const
{
  SO2 result(*this);
  result.unit_complex_ *= other.unit_complex_;
  result.normalize();
  return result;
}

void SO2
::operator*=(const SO2& other)
{
  unit_complex_ *= other.unit_complex_;
  normalize();
}

Vector2d SO2
::operator*(const Vector2d & xy) const
{
  const double & real = unit_complex_.real();
  const double & imag = unit_complex_.imag();
  return Vector2d(real*xy[0] - imag*xy[1], imag*xy[0] + real*xy[1]);
}

SO2 SO2
::inverse() const
{
  return SO2(conj(unit_complex_));
}

Matrix2d SO2
::matrix() const
{
  const double & real = unit_complex_.real();
  const double & imag = unit_complex_.imag();
  Matrix2d R;
  R(0,0) = real;  R(0,1) = -imag;
  R(1,0) = imag; R(1,1) = real;
  return R;
}

double SO2
::Adj() const
{
  return 1.;
}

Matrix2d SO2
::generator(int i)
{
  assert(i==0);
  return hat(1.);
}

double SO2
::log() const
{
  return SO2::log(*this);
}

double SO2
::log(const SO2 & other)
{
  return atan2(other.unit_complex_.imag(), other.unit_complex().real());
}

SO2 SO2
::exp(double theta)
{
  return SO2(Complexd(cos(theta), sin(theta)));
}

Matrix2d SO2
::hat(double v)
{
  Matrix2d Omega;
  Omega <<  0, -v
      ,     v,  0;
  return Omega;
}

double SO2
::vee(const Matrix2d & Omega)
{
  assert(fabs(Omega(1,0)+Omega(0,1))<SMALL_EPS);
  return Omega(1,0);
}

double SO2
::lieBracket(double omega1, double omega2)
{
  return 0.;
}

void SO2::
setComplex(const Complexd& complex)
{
  assert(abs(complex)!=0);
  unit_complex_ = complex;
  normalize();
}

void SO2::
normalize()
{
  unit_complex_ /= abs(unit_complex_);
}


}
