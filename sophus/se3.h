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

#ifndef SOPHUS_SE3_H
#define SOPHUS_SE3_H

#include "so3.h"

namespace Sophus
{
using namespace Eigen;
using namespace std;

////////////////////////////////////////////////////////////////////////////
// Forward Declarations / typedefs
////////////////////////////////////////////////////////////////////////////

template<typename Scalar> class SE3Group;
typedef SE3Group<double> SE3;

typedef Matrix<double,6,1> Vector6d;
typedef Matrix<double,6,6> Matrix6d;

////////////////////////////////////////////////////////////////////////////
// Template Definition
////////////////////////////////////////////////////////////////////////////

template<typename Scalar=double>
class SE3Group
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SE3Group                        ();

  SE3Group                        (const SO3Group<Scalar> & so3,
                              const Matrix<Scalar,3,1> & translation);
  SE3Group                        (const Matrix3d & rotation_matrix,
                              const Matrix<Scalar,3,1> & translation);
  SE3Group                        (const Quaternion<Scalar> & unit_quaternion,
                              const Matrix<Scalar,3,1> & translation_);
  SE3Group                        (const Eigen::Matrix<Scalar,4,4>& T);
  SE3Group                        (const SE3Group & other);

  SE3Group &
  operator=                  (const SE3Group & other);

  SE3Group
  operator*                  (const SE3Group& other) const;

  SE3Group&
  operator*=                 (const SE3Group& other);

  SE3Group
  inverse                    () const;

  Matrix<Scalar,6,1>
  log                        () const;

  Matrix<Scalar,3,1>
  operator*                  (const Matrix<Scalar,3,1> & xyz) const;

  static SE3Group
  exp                        (const Matrix<Scalar,6,1> & update);

  static Matrix<Scalar,6,1>
  log                        (const SE3Group & SE3Group);

  Matrix<Scalar,3,4>
  matrix3x4                  () const;

  Matrix<Scalar,4,4>
  matrix                     () const;

  Matrix<Scalar, 6, 6>
  Adj                        () const;

  static Matrix<Scalar,4,4>
  hat                        (const Matrix<Scalar,6,1> & omega);

  static Matrix<Scalar,6,1>
  vee                        (const Matrix<Scalar,4,4> & Omega);

  static Matrix<Scalar,6,1>
  lieBracket                 (const Matrix<Scalar,6,1> & v1,
                              const Matrix<Scalar,6,1> & v2);

  static Matrix<Scalar,6,6>
  d_lieBracketab_by_d_a      (const Matrix<Scalar,6,1> & b);

  void
  setQuaternion              (const Quaternion<Scalar>& quaternion);

  const Matrix<Scalar,3,1> & translation() const
  {
    return translation_;
  }

  Matrix<Scalar,3,1>& translation()
  {
    return translation_;
  }

  const Quaternion<Scalar> & unit_quaternion() const
  {
    return so3_.unit_quaternion();
  }

  Matrix3d rotation_matrix() const
  {
    return so3_.matrix();
  }

  void setRotationMatrix(const Matrix3d & rotation_matrix)
  {
    so3_.setQuaternion(Quaternion<Scalar>(rotation_matrix));
  }

  const SO3Group<Scalar>& so3() const
  {
    return so3_;
  }

  SO3Group<Scalar>& so3()
  {
    return so3_;
  }

  static const int DoF = 6;

private:
  SO3Group<Scalar> so3_;
  Matrix<Scalar,3,1> translation_;

};

////////////////////////////////////////////////////////////////////////////
// Template Implementation
////////////////////////////////////////////////////////////////////////////

template<typename Scalar> inline
SE3Group<Scalar>
::SE3Group()
{
  translation_.setZero();
}

template<typename Scalar> inline
SE3Group<Scalar>
::SE3Group(const SO3Group<Scalar> & so3, const Matrix<Scalar,3,1> & translation)
  : so3_(so3), translation_(translation) {}

template<typename Scalar> inline
SE3Group<Scalar>
::SE3Group(const Matrix3d & rotation_matrix, const Matrix<Scalar,3,1> & translation)
  : so3_(rotation_matrix), translation_(translation){}

template<typename Scalar> inline
SE3Group<Scalar>
::SE3Group(const Quaternion<Scalar> & quaternion, const Matrix<Scalar,3,1> & translation)
  : so3_(quaternion), translation_(translation) {}

template<typename Scalar> inline
SE3Group<Scalar>
::SE3Group(const Eigen::Matrix<Scalar,4,4>& T)
  : so3_(T.template topLeftCorner<3,3>()), translation_(T.template block<3,1>(0,3)) {}

template<typename Scalar> inline
SE3Group<Scalar>
::SE3Group(const SE3Group & SE3Group) : so3_(SE3Group.so3_),translation_(SE3Group.translation_){}


template<typename Scalar> inline
SE3Group<Scalar> & SE3Group<Scalar>
::operator = (const SE3Group & other)
{
  so3_ = other.so3_;
  translation_ = other.translation_;
  return *this;
}

template<typename Scalar> inline
SE3Group<Scalar> SE3Group<Scalar>
::operator*(const SE3Group & other) const
{
  SE3Group result(*this);
  result.translation_ += so3_*(other.translation_);
  result.so3_*=other.so3_;
  return result;
}

template<typename Scalar> inline
SE3Group<Scalar>& SE3Group<Scalar>
::operator *= (const SE3Group & other)
{
  translation_+= so3_*(other.translation_);
  so3_*=other.so3_;
  return *this;
}

template<typename Scalar> inline
SE3Group<Scalar> SE3Group<Scalar>
::inverse() const
{
  SE3Group ret;
  ret.so3_= so3_.inverse();
  ret.translation_= ret.so3_*(translation_*-1.);
  return ret;
}

template<typename Scalar> inline
Matrix<Scalar,6,1> SE3Group<Scalar>
::log() const
{
  return log(*this);
}

template<typename Scalar> inline
Matrix<Scalar,3,1> SE3Group<Scalar>
::operator *(const Matrix<Scalar,3,1> & xyz) const
{
  return so3_*xyz + translation_;
}

template<typename Scalar> inline
Matrix<Scalar,3,4> SE3Group<Scalar>
::matrix3x4() const
{
  Matrix<Scalar,3,4> matrix;
  matrix.block(0,0,3,3) = rotation_matrix();
  matrix.col(3) = translation_;
  return matrix;
}

template<typename Scalar> inline
Matrix<Scalar,4,4> SE3Group<Scalar>
::matrix() const
{
  Matrix<Scalar,4,4> homogenious_matrix;
  homogenious_matrix.setIdentity();
  homogenious_matrix.block(0,0,3,3) = rotation_matrix();
  homogenious_matrix.col(3).head(3) = translation_;
  return homogenious_matrix;
}


template<typename Scalar> inline
Matrix<Scalar, 6, 6> SE3Group<Scalar>
::Adj() const
{
  Matrix3d R = so3_.matrix();
  Matrix<Scalar, 6, 6> res;
  res.block(0,0,3,3) = R;
  res.block(3,3,3,3) = R;
  res.block(0,3,3,3) = SO3Group<Scalar>::hat(translation_)*R;
  res.block(3,0,3,3) = Matrix3d::Zero(3,3);
  return res;
}

template<typename Scalar> inline
Matrix<Scalar,4,4> SE3Group<Scalar>
::hat(const Matrix<Scalar,6,1> & v)
{
  Matrix<Scalar,4,4> Omega;
  Omega.setZero();
  Omega.template topLeftCorner<3,3>() = SO3Group<Scalar>::hat(v.template tail<3>());
  Omega.col(3).template head<3>() = v.template head<3>();
  return Omega;
}

template<typename Scalar> inline
Matrix<Scalar,6,1> SE3Group<Scalar>
::vee(const Matrix<Scalar,4,4> & Omega)
{
  Matrix<Scalar,6,1> upsilon_omega;
  upsilon_omega.template head<3>() = Omega.col(3).template head<3>();
  upsilon_omega.template tail<3>() = SO3Group<Scalar>::vee(Omega.template topLeftCorner<3,3>());
  return upsilon_omega;
}

template<typename Scalar> inline
Matrix<Scalar,6,1> SE3Group<Scalar>
::lieBracket(const Matrix<Scalar,6,1> & v1, const Matrix<Scalar,6,1> & v2)
{
  Matrix<Scalar,3,1> upsilon1 = v1.template head<3>();
  Matrix<Scalar,3,1> upsilon2 = v2.template head<3>();
  Matrix<Scalar,3,1> omega1 = v1.template tail<3>();
  Matrix<Scalar,3,1> omega2 = v2.template tail<3>();

  Matrix<Scalar,6,1> res;
  res.template head<3>() = omega1.cross(upsilon2) + upsilon1.cross(omega2);
  res.template tail<3>() = omega1.cross(omega2);

  return res;
}

template<typename Scalar> inline
Matrix<Scalar,6,6> SE3Group<Scalar>
::d_lieBracketab_by_d_a(const Matrix<Scalar,6,1> & b)
{
  Matrix<Scalar,6,6> res;
  res.setZero();

  Matrix<Scalar,3,1> upsilon2 = b.template head<3>();
  Matrix<Scalar,3,1> omega2 = b.template tail<3>();

  res.template topLeftCorner<3,3>() = -SO3Group<Scalar>::hat(omega2);
  res.template topRightCorner<3,3>() = -SO3Group<Scalar>::hat(upsilon2);

  res.template bottomRightCorner<3,3>() = -SO3Group<Scalar>::hat(omega2);
  return res;
}

template<typename Scalar> inline
SE3Group<Scalar> SE3Group<Scalar>
::exp(const Matrix<Scalar,6,1> & update)
{
  Matrix<Scalar,3,1> upsilon = update.template head<3>();
  Matrix<Scalar,3,1> omega = update.template tail<3>();

  Scalar theta;
  SO3Group<Scalar> so3 = SO3Group<Scalar>::expAndTheta(omega, &theta);

  Matrix3d Omega = SO3Group<Scalar>::hat(omega);
  Matrix3d Omega_sq = Omega*Omega;
  Matrix3d V;

  if(theta<SMALL_EPS)
  {
    V = so3.matrix();
    //Note: That is an accurate expansion!
  }
  else
  {
    Scalar theta_sq = theta*theta;
    V = (Matrix3d::Identity()
         + (1-cos(theta))/(theta_sq)*Omega
         + (theta-sin(theta))/(theta_sq*theta)*Omega_sq);
  }
  return SE3Group(so3,V*upsilon);
}

template<typename Scalar> inline
Matrix<Scalar,6,1> SE3Group<Scalar>
::log(const SE3Group & SE3Group)
{
  Matrix<Scalar,6,1> upsilon_omega;
  Scalar theta;
  upsilon_omega.template tail<3>() = SO3Group<Scalar>::logAndTheta(SE3Group.so3_, &theta);

  if (theta<SMALL_EPS)
  {
    Matrix3d Omega = SO3Group<Scalar>::hat(upsilon_omega.template tail<3>());
    Matrix3d V_inv = Matrix3d::Identity()- 0.5*Omega + (1./12.)*(Omega*Omega);

    upsilon_omega.template head<3>() = V_inv*SE3Group.translation_;
  }
  else
  {
    Matrix3d Omega = SO3Group<Scalar>::hat(upsilon_omega.template tail<3>());
    Matrix3d V_inv = ( Matrix3d::Identity() - 0.5*Omega
              + ( 1-theta/(2*tan(theta/2)))/(theta*theta)*(Omega*Omega) );
    upsilon_omega.template head<3>() = V_inv*SE3Group.translation_;
  }
  return upsilon_omega;
}

template<typename Scalar> inline
void SE3Group<Scalar>::
setQuaternion(const Quaternion<Scalar>& quat)
{
  return so3_.setQuaternion(quat);
}


} // end namespace


#endif
