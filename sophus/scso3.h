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

#ifndef SOPHUS_ScSO3_H
#define SOPHUS_ScSO3_H

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Geometry>

#include "so3.h"


namespace Sophus
{
using namespace Eigen;

class ScSO3
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScSO3                      ();

  ScSO3                      (const ScSO3 & other);

  explicit
  ScSO3                      (const Matrix3d & scale_times_R);

  ScSO3                      (double scale,
                              const SO3 & so3);

  ScSO3                      (double scale,
                              const Matrix3d & R);

  explicit
  ScSO3                      (const Quaterniond & quaternion);

  void
  operator=                  (const ScSO3 & ScSO3);

  ScSO3
  operator*                  (const ScSO3 & ScSO3) const;

  void
  operator*=                 (const ScSO3 & ScSO3);

  Vector3d
  operator*                  (const Vector3d & xyz) const;

  ScSO3
  inverse                    () const;

  Matrix3d
  matrix                     () const;

  Matrix3d
  rotationMatrix             () const;

  double
  scale                      () const;

  void
  setRotationMatrix          (const Matrix3d & R);

  void
  setScale                   (double scale);

  Matrix4d
  Adj                        () const;

  Matrix3d
  generator                  (int i);

  Vector4d
  log                        () const;

  static ScSO3
  exp                        (const Vector4d & omega);

  static ScSO3
  expAndTheta                (const Vector4d & omega,
                              double * theta);

  static Vector4d
  log                        (const ScSO3 & ScSO3);

  static Vector4d
  logAndTheta                (const ScSO3 & ScSO3,
                              double * theta);

  static Matrix3d
  hat                        (const Vector4d & omega);

  static Vector4d
  vee                        (const Matrix3d & Omega);

  static Vector4d
  lieBracket                 (const Vector4d & omega1,
                              const Vector4d & omega2);

  static Matrix4d
  d_lieBracketab_by_d_a      (const Vector4d & b);


  const Quaterniond & quaternion() const
  {
    return quaternion_;
  }
  Quaterniond& quaternion()
  {
    return quaternion_;
  }

  static const int DoF = 4;

protected:
  Quaterniond quaternion_;
};

inline std::ostream& operator <<(std::ostream & out_str,
                                 const ScSO3& scso3)
{
  out_str << scso3.scale() << " * " <<
             scso3.log().head<3>().transpose() << std::endl;
  return out_str;
}

} // end namespace


#endif
