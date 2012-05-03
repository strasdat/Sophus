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

#ifndef SOPHUS_SE2_H
#define SOPHUS_SE2_H

#include "so2.h"

namespace Sophus
{
using namespace Eigen;
using namespace std;

class SE2
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SE2                        ();

  SE2                        (const SO2 & so2,
                              const Vector2d & translation);

  SE2                        (const Matrix2d & rotation_matrix,
                              const Vector2d & translation);

  SE2                        (double theta,
                              const Vector2d & translation_);

  SE2                        (const SE2 & other);

  SE2 &
  operator=                  (const SE2 & other);

  SE2
  operator*                  (const SE2& other) const;

  SE2&
  operator*=                 (const SE2& other);

  SE2
  inverse                    () const;

  Vector3d
  log                        () const;

  Vector2d
  operator*                  (const Vector2d & xy) const;

  static SE2
  exp                        (const Vector3d & update);

  static Vector3d
  log                        (const SE2 & SE2);

  Matrix<double,3,3>
  matrix                     () const;

  Matrix<double, 3, 3>
  Adj                        () const;

  static Matrix3d
  hat                        (const Vector3d & omega);

  static Vector3d
  vee                        (const Matrix3d & Omega);

  static Vector3d
  lieBracket                 (const Vector3d & v1,
                              const Vector3d & v2);

  static Matrix3d
  d_lieBracketab_by_d_a      (const Vector3d & b);

  const Vector2d & translation() const
  {
    return translation_;
  }

  Vector2d& translation()
  {
    return translation_;
  }

  Matrix2d rotation_matrix() const
  {
    return so2_.matrix();
  }

  void setRotationMatrix(const Matrix2d & rotation_matrix)
  {
    so2_ = SO2(rotation_matrix);
  }

  const SO2& so2() const
  {
    return so2_;
  }

  SO2& so2()
  {
    return so2_;
  }

  static const int DoF = 3;

private:
  SO2 so2_;
  Vector2d translation_;

};

inline std::ostream& operator <<(std::ostream & out_str,
                                 const SE2 &  SE2)
{
  out_str << SE2.so2() << SE2.translation() << std::endl;
  return out_str;
}

} // end namespace


#endif
