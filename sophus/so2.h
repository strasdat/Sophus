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

#ifndef SOPHUS_SO2_H
#define SOPHUS_SO2_H

#include <complex>

#include <Eigen/Core>


namespace Sophus
{
using namespace Eigen;


class SO2
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::complex<double> Complexd;

  SO2                        ();

  SO2                        (const SO2 & other);

  explicit
  SO2                        (const Matrix2d & R);

  explicit
  SO2                        (const Complexd & unit_complex);

  SO2                        (double theta);
  void
  operator=                  (const SO2 & SO2);

  SO2
  operator*                  (const SO2 & SO2) const;

  void
  operator*=                 (const SO2 & SO2);

  Vector2d
  operator*                  (const Vector2d & xy) const;

  SO2
  inverse                    () const;

  Matrix2d
  matrix                     () const;

  double
  Adj                        () const;

  Matrix2d
  generator                  (int i);

  double
  log                        () const;

  static SO2
  exp                        (double theta);

  static double
  log                        (const SO2 & SO2);

  static Matrix2d
  hat                        (double omega);

  static double
  vee                        (const Matrix2d & Omega);

  static double
  lieBracket                 (double omega1,
                              double omega2);
  void
  setComplex                 (const Complexd& z);

  const Complexd & unit_complex() const
  {
    return unit_complex_;
  }

  static const int DoF = 2;

protected:
  void
  normalize                  ();

  Complexd unit_complex_;
};

inline std::ostream& operator <<(std::ostream & out_str,
                                 const SO2 & SO2)
{

  out_str << SO2.log() << std::endl;
  return out_str;
}

} // end namespace


#endif
