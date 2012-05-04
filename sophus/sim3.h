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

#ifndef SOPHUS_SIMILARITY_H
#define SOPHUS_SIMILARITY_H

#include "scso3.h"
#include "se3.h"

namespace Sophus
{
typedef Matrix< double, 7, 1 > Vector7d;
typedef Matrix< double, 7, 7 > Matrix7d;

/**
   * This class implements the Lie group and the corresponding Lie
   * algebra of 3D similarity transformationns Sim3 as described in:
   *
   * > H. Strasdat, J.M.M. Montiel, A.J. Davison:
   *   "Scale Drift-Aware Large Scale Monocular SLAM",
   *   Proc. of Robotics: Science and Systems (RSS),
   *   Zaragoza, Spain, 2010.
   *   http://www.roboticsproceedings.org/rss06/p10.html
   */
class Sim3
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Sim3                       ();

  Sim3                       (const ScSO3& scso3,
                              const Vector3d & t);
  Sim3                       (double s,
                              const Matrix3d & R,
                              const Vector3d & t);
  Sim3                       (const Quaterniond& q,
                              const Vector3d & t);
  Sim3                       (const Sim3 & sim3);

  static Sim3
  from_SE3                   (const SE3 & se3);

  static SE3
  to_SE3                     (const Sim3 & sim3);

  SE3
  to_SE3                     () const;

  Matrix7d
  Adj                        () const;

  static Sim3
  exp                        (const Matrix<double,7,1> & vect);

  static Vector7d
  log                        (const Sim3& sim3);

  Sim3
  inverse                    () const;

  Sim3 &
  operator*=                 (const Sim3 & sim3);

  Sim3
  operator*                  (const Sim3 & sim3) const;

  Matrix4d
  matrix                     () const;

  Vector3d
  operator*                  (const Vector3d & xyz) const;

  static Matrix4d
  hat                        (const Vector7d & omega);

  static Vector7d
  vee                        (const Matrix4d & Omega);

  static Vector7d
  lieBracket                 (const Vector7d & omega1,
                              const Vector7d & omega2);

  static Matrix7d
  d_lieBracketab_by_d_a      (const Vector7d & b);

  SO3
  so3                        () const;

  const Vector3d& translation() const
  {
    return translation_;
  }

  Vector3d& translation()
  {
    return translation_;
  }

  const Quaterniond & quaternion() const
  {
    return scso3_.quaternion();
  }

  Quaterniond& quaternion()
  {
    return scso3_.quaternion();
  }

  const ScSO3 & scso3() const
  {
    return scso3_;
  }

  ScSO3& scso3()
  {
    return scso3_;
  }

  Matrix3d rotation_matrix() const
  {
    return scso3_.rotationMatrix();
  }


  double scale() const
  {
    return scso3_.scale();
  }

  Matrix<double,7,1> log() const
  {
    return Sim3::log(*this);
  }

  static const int DoF = 7;

protected:
  static Matrix3d
  calcW                      (double theta,
                              double sigma,
                              double scale,
                              const Matrix3d & Omega);

  ScSO3 scso3_;
  Vector3d translation_;
};


inline std::ostream& operator <<(std::ostream& out_str,
                                 const Sim3 &  sim3)
{
  out_str << sim3.scso3() <<
             sim3.translation().transpose() << std::endl;
  return out_str;
}


}


#endif
