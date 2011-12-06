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

  Sim3                       (const SO3& so3,
                              const Vector3d & t,
                              double s);
  Sim3                       (const Matrix3d & R,
                              const Vector3d & t,
                              double s);
  Sim3                       (const Quaterniond& q,
                              const Vector3d & t,
                              double s);
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

  const Vector3d& translation() const
  {
    return translation_;
  }

  Vector3d& translation()
  {
    return translation_;
  }

  const Quaterniond & quat() const
  {
    return so3_.quaternion();
  }

  Quaterniond& quat()
  {
    return so3_.quaternion();
  }

  Matrix3d rotation_matrix() const
  {
    return so3_.matrix();
  }

  void set_rotation_matrix(const Matrix3d & R)
  {
    so3_.quaternion()= Quaterniond(R);
  }

  double& scale()
  {
    return scale_;
  }

  const double& scale() const
  {
    return scale_;
  }


  inline const SO3& so3() const
  {
    return so3_;
  }

  inline SO3& so3()
  {
    return so3_;
  }

  Matrix<double,7,1> log() const
  {
    return Sim3::log(*this);
  }

  //TODO: Yes we assume at the moment a fixed scale as in the ICCV paper
  //      I'll reconsider this problem later...
  static const int DoF = 6;

protected:
  SO3 so3_;
  Vector3d translation_;
  double scale_;
};


inline std::ostream& operator <<(std::ostream& out_str,
                                 const Sim3 &  sim3)
{
  Matrix<double,4,4> homogenious_matrix;
  homogenious_matrix.setIdentity();
  homogenious_matrix.block(0,0,3,3) = sim3.rotation_matrix();
  homogenious_matrix.col(3).head(3) = sim3.translation();
  homogenious_matrix(3,3) = 1;
  out_str << sim3.scale() << " * " << endl <<
             homogenious_matrix << std::endl;

  return out_str;
}


}


#endif
