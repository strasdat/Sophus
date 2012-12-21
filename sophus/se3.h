// This file is part of Sophus.
//
// Copyright 2011 Hauke Strasdat (Imperial College London)
//           2012 Steven Lovegrove, Hauke Strasdat
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

#include <iostream>
#include "so3.h"

////////////////////////////////////////////////////////////////////////////
// Forward Declarations / typedefs
////////////////////////////////////////////////////////////////////////////

namespace Sophus {
template<typename _Scalar, int _Options=0> class SE3Group;
typedef SE3Group<double> SE3;
typedef Matrix<double,6,1> Vector6d;
typedef Matrix<double,6,6> Matrix6d;
}

////////////////////////////////////////////////////////////////////////////
// Eigen Traits (For querying derived types in CRTP hierarchy)
////////////////////////////////////////////////////////////////////////////

namespace Eigen {
namespace internal {

template<typename _Scalar, int _Options>
struct traits<Sophus::SE3Group<_Scalar,_Options> > {
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,3,1> TranslationType;
  typedef Sophus::SO3Group<Scalar> SO3Type;
};

template<typename _Scalar, int _Options>
struct traits<Map<Sophus::SE3Group<_Scalar>, _Options> >
    : traits<Sophus::SE3Group<_Scalar, _Options> > {
  typedef _Scalar Scalar;
  typedef Map<Matrix<Scalar,3,1>,_Options> TranslationType;
  typedef Map<Sophus::SO3Group<Scalar>,_Options> SO3Type;
};

template<typename _Scalar, int _Options>
struct traits<Map<const Sophus::SE3Group<_Scalar>, _Options> >
    : traits<const Sophus::SE3Group<_Scalar, _Options> > {
  typedef _Scalar Scalar;
  typedef Map<const Matrix<Scalar,3,1>,_Options> TranslationType;
  typedef Map<const Sophus::SO3Group<Scalar>,_Options> SO3Type;
};

}
}



namespace Sophus {
using namespace Eigen;
using namespace std;

////////////////////////////////////////////////////////////////////////////
// SE3GroupBase type - implements SE3 class but is storage agnostic
////////////////////////////////////////////////////////////////////////////

template<typename Derived>
class SE3GroupBase {
public:
  typedef typename internal::traits<Derived>::Scalar Scalar;
  typedef typename internal::traits<Derived>::TranslationType TranslationType;
  typedef typename internal::traits<Derived>::SO3Type SO3Type;
  static const int DoF = 6;

  inline
  SE3GroupBase<Derived>& operator = (const SE3Group<Scalar> & other) {
    so3() = other.so3();
    translation() = other.translation();
    return *this;
  }

  inline
  const SE3Group<Scalar> operator*(const SE3Group<Scalar> & other) const {
    SE3Group<Scalar> result(*this);
    result.translation() += so3()*(other.translation());
    result.so3()*=other.so3();
    return result;
  }

  inline
  const SE3Group<Scalar>& operator *= (const SE3Group<Scalar> & other) {
    translation()+= so3()*(other.translation());
    so3()*=other.so3();
    return *this;
  }

  inline
  const SE3Group<Scalar> inverse() const {
    const SO3Group<Scalar> invR = so3().inverse();
    return SE3Group<Scalar>(invR, invR*(translation()
                                        *static_cast<Scalar>(-1) ) );
  }

  inline
  const Matrix<Scalar,6,1> log() const {
    return log(*this);
  }

  inline
  const Matrix<Scalar,3,1> operator *(const Matrix<Scalar,3,1> & xyz) const {
    return so3()*xyz + translation();
  }

  inline
  const Matrix<Scalar,3,4> matrix3x4() const {
    Matrix<Scalar,3,4> matrix;
    matrix.block(0,0,3,3) = rotation_matrix();
    matrix.col(3) = translation();
    return matrix;
  }

  inline
  const Matrix<Scalar,4,4> matrix() const {
    Matrix<Scalar,4,4> homogenious_matrix;
    homogenious_matrix.setIdentity();
    homogenious_matrix.block(0,0,3,3) = rotation_matrix();
    homogenious_matrix.col(3).head(3) = translation();
    return homogenious_matrix;
  }

  inline
  const Matrix<Scalar, 6, 6> Adj() const {
    Matrix3d R = so3().matrix();
    Matrix<Scalar, 6, 6> res;
    res.block(0,0,3,3) = R;
    res.block(3,3,3,3) = R;
    res.block(0,3,3,3) = SO3Group<Scalar>::hat(translation())*R;
    res.block(3,0,3,3) = Matrix3d::Zero(3,3);
    return res;
  }

  inline static
  const Matrix<Scalar,4,4> hat(const Matrix<Scalar,6,1> & v) {
    Matrix<Scalar,4,4> Omega;
    Omega.setZero();
    Omega.template topLeftCorner<3,3>()
        = SO3Group<Scalar>::hat(v.template tail<3>());
    Omega.col(3).template head<3>() = v.template head<3>();
    return Omega;
  }

  inline static
  const Matrix<Scalar,6,1> vee(const Matrix<Scalar,4,4> & Omega) {
    Matrix<Scalar,6,1> upsilon_omega;
    upsilon_omega.template head<3>() = Omega.col(3).template head<3>();
    upsilon_omega.template tail<3>()
        = SO3Group<Scalar>::vee(Omega.template topLeftCorner<3,3>());
    return upsilon_omega;
  }

  inline static
  const Matrix<Scalar,6,1> lieBracket(const Matrix<Scalar,6,1> & v1,
                                const Matrix<Scalar,6,1> & v2) {
    Matrix<Scalar,3,1> upsilon1 = v1.template head<3>();
    Matrix<Scalar,3,1> upsilon2 = v2.template head<3>();
    Matrix<Scalar,3,1> omega1 = v1.template tail<3>();
    Matrix<Scalar,3,1> omega2 = v2.template tail<3>();

    Matrix<Scalar,6,1> res;
    res.template head<3>() = omega1.cross(upsilon2) + upsilon1.cross(omega2);
    res.template tail<3>() = omega1.cross(omega2);

    return res;
  }

  inline static
  const Matrix<Scalar,6,6> d_lieBracketab_by_d_a(const Matrix<Scalar,6,1> & b) {
    Matrix<Scalar,6,6> res;
    res.setZero();

    Matrix<Scalar,3,1> upsilon2 = b.template head<3>();
    Matrix<Scalar,3,1> omega2 = b.template tail<3>();

    res.template topLeftCorner<3,3>() = -SO3Group<Scalar>::hat(omega2);
    res.template topRightCorner<3,3>() = -SO3Group<Scalar>::hat(upsilon2);

    res.template bottomRightCorner<3,3>() = -SO3Group<Scalar>::hat(omega2);
    return res;
  }

  inline static
  const SE3Group<Scalar> exp(const Matrix<Scalar,6,1> & update) {
    Matrix<Scalar,3,1> upsilon = update.template head<3>();
    Matrix<Scalar,3,1> omega = update.template tail<3>();

    Scalar theta;
    SO3Group<Scalar> so3 = SO3Group<Scalar>::expAndTheta(omega, &theta);

    Matrix<Scalar,3,3> Omega = SO3Group<Scalar>::hat(omega);
    Matrix<Scalar,3,3> Omega_sq = Omega*Omega;
    Matrix<Scalar,3,3> V;

    if(theta<SophusConstants<Scalar>::epsilon()) {
      V = so3.matrix();
      //Note: That is an accurate expansion!
    } else {
      Scalar theta_sq = theta*theta;
      V = (Matrix<Scalar,3,3>::Identity()
           + (static_cast<Scalar>(1)-cos(theta))/(theta_sq)*Omega
           + (theta-sin(theta))/(theta_sq*theta)*Omega_sq);
    }
    return SE3Group<Scalar>(so3,V*upsilon);
  }

  inline static
  const Matrix<Scalar,6,1> log(const SE3Group<Scalar> & se3) {
    Matrix<Scalar,6,1> upsilon_omega;
    Scalar theta;
    upsilon_omega.template tail<3>()
        = SO3Group<Scalar>::logAndTheta(se3.so3(), &theta);

    if (fabs(theta)<SophusConstants<Scalar>::epsilon()) {
      const Matrix<Scalar,3,3> Omega
          = SO3Group<Scalar>::hat(upsilon_omega.template tail<3>());
      const Matrix<Scalar,3,3> V_inv =
          Matrix<Scalar,3,3>::Identity() -
          static_cast<Scalar>(0.5)*Omega
          + static_cast<Scalar>(1./12.)*(Omega*Omega);

      upsilon_omega.template head<3>() = V_inv*se3.translation();
    } else {
      const Matrix<Scalar,3,3> Omega
          = SO3Group<Scalar>::hat(upsilon_omega.template tail<3>());
      const Matrix<Scalar,3,3> V_inv =
          ( Matrix<Scalar,3,3>::Identity() - static_cast<Scalar>(0.5)*Omega
            + ( static_cast<Scalar>(1)
                - theta/(static_cast<Scalar>(2)*tan(theta/Scalar(2)))) /
            (theta*theta)*(Omega*Omega) );
      upsilon_omega.template head<3>() = V_inv*se3.translation();
    }
    return upsilon_omega;
  }

  // GETTERS & SETTERS

  EIGEN_STRONG_INLINE
  const TranslationType& translation() const {
      return static_cast<const Derived*>(this)->translation();
  }

  EIGEN_STRONG_INLINE
  const SO3Type& so3() const {
      return static_cast<const Derived*>(this)->so3();
  }

  EIGEN_STRONG_INLINE
  TranslationType& translation() {
      return static_cast<Derived*>(this)->translation();
  }

  EIGEN_STRONG_INLINE
  SO3Type& so3() {
      return static_cast<Derived*>(this)->so3();
  }


  inline
  void setQuaternion(const typename SO3Type::QuaternionType& quat) {
    return so3().setQuaternion(quat);
  }

  inline
  const typename SO3Type::QuaternionType& unit_quaternion() const {
    return so3().unit_quaternion();
  }

  inline
  const Matrix<Scalar,3,3> rotation_matrix() const {
    return so3().matrix();
  }

  inline
  void setRotationMatrix(const Matrix3d & rotation_matrix) {
    so3().setQuaternion(SO3Type::QuaternionType(rotation_matrix));
  }

  template<typename NewScalarType>
  inline SE3Group<NewScalarType> cast() const {
    return
        SE3Group<NewScalarType>(so3().template cast<NewScalarType>(),
                                translation().template cast<NewScalarType>() );
  }

};

////////////////////////////////////////////////////////////////////////////
// SE3Group type - Constructors and default storage for SE3 Type
////////////////////////////////////////////////////////////////////////////

template<typename _Scalar, int _Options>
class SE3Group : public SE3GroupBase<SE3Group<_Scalar,_Options> >
{
public:
  typedef typename internal::traits<SE3Group<_Scalar,_Options> >
  ::Scalar Scalar;
  typedef typename internal::traits<SE3Group<_Scalar,_Options> >
  ::TranslationType TranslationType;
  typedef typename internal::traits<SE3Group<_Scalar,_Options> >
  ::SO3Type SO3Type;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline
  SE3Group()
  {
    translation_.setZero();
  }

  template<typename OtherDerived> inline
  SE3Group(const SO3GroupBase<OtherDerived> & so3,
           const Matrix<Scalar,3,1> & translation)
    : translation_(translation), so3_(so3) {
  }

  inline
  SE3Group(const Matrix3d & rotation_matrix,
           const Matrix<Scalar,3,1> & translation)
    : translation_(translation), so3_(rotation_matrix) {
  }

  inline
  SE3Group(const Quaternion<Scalar> & quaternion,
           const Matrix<Scalar,3,1> & translation)
    : translation_(translation), so3_(quaternion) {
  }

  inline
  SE3Group(const Eigen::Matrix<Scalar,4,4>& T)
    : translation_(T.template block<3,1>(0,3)),
      so3_(T.template topLeftCorner<3,3>()) {
  }

  template<typename OtherDerived> inline
  SE3Group(const SE3GroupBase<OtherDerived> & other)
    : translation_(other.translation()), so3_(other.so3()) {
  }

  // GETTERS & SETTERS

  EIGEN_STRONG_INLINE
  const TranslationType& translation() const {
    return translation_;
  }

  EIGEN_STRONG_INLINE
  const SO3Type& so3() const {
    return so3_;
  }

  EIGEN_STRONG_INLINE
  TranslationType& translation() {
    return translation_;
  }

  EIGEN_STRONG_INLINE
  SO3Type& so3() {
    return so3_;
  }

  EIGEN_STRONG_INLINE
  Scalar* data()
  {
    // TODO: Check this is true
    // translation_ and so3_ are layed out sequentially with no padding
    return translation_.data();
  }

  EIGEN_STRONG_INLINE
  const Scalar* data() const
  {
    // TODO: Check this is true
    // translation_ and so3_ are layed out sequentially with no padding
    return translation_.data();
  }

protected:
  TranslationType translation_;
  SO3Type so3_;
};


} // end namespace

////////////////////////////////////////////////////////////////////////////
// Specialisation of Eigen::Map for SE3GroupBase
// Allows us to wrap SE3 Objects around POD array
// (e.g. external c style xyz vector + quaternion)
////////////////////////////////////////////////////////////////////////////

namespace Eigen {

template<typename _Scalar, int _Options>
class Map<Sophus::SE3Group<_Scalar>, _Options>
    : public Sophus::SE3GroupBase<Map<Sophus::SE3Group<_Scalar>, _Options> >
{
  typedef Sophus::SE3GroupBase<Map<Sophus::SE3Group<_Scalar>, _Options> > Base;

public:
  typedef typename internal::traits<Map>::Scalar Scalar;
  typedef typename internal::traits<Map>::TranslationType TranslationType;
  typedef typename internal::traits<Map>::SO3Type SO3Type;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  EIGEN_STRONG_INLINE
  Map(Scalar* coeffs) : translation_(coeffs), so3_(coeffs+3) {
  }

  // GETTERS & SETTERS

  EIGEN_STRONG_INLINE
  const TranslationType& translation() const {
    return translation_;
  }

  EIGEN_STRONG_INLINE
  const SO3Type& so3() const {
    return so3_;
  }

  EIGEN_STRONG_INLINE
  TranslationType& translation() {
    return translation_;
  }

  EIGEN_STRONG_INLINE
  SO3Type& so3() {
    return so3_;
  }

protected:
  TranslationType translation_;
  SO3Type so3_;
};

template<typename _Scalar, int _Options>
class Map<const Sophus::SE3Group<_Scalar>, _Options>
    : public Sophus::SE3GroupBase<
    Map<const Sophus::SE3Group<_Scalar>, _Options> > {
  typedef Sophus::SE3GroupBase<Map<const Sophus::SE3Group<_Scalar>, _Options> >
  Base;

public:
  typedef typename internal::traits<Map>::Scalar Scalar;
  typedef typename internal::traits<Map>::TranslationType TranslationType;
  typedef typename internal::traits<Map>::SO3Type SO3Type;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  EIGEN_STRONG_INLINE
  Map(const Scalar* coeffs)
    : translation_(coeffs), so3_(coeffs+3){
  }

  EIGEN_STRONG_INLINE
  Map(const Scalar* trans_coeffs, const Scalar* rot_coeffs)
    : translation_(trans_coeffs), so3_(rot_coeffs){
  }

  // GETTERS & SETTERS

  EIGEN_STRONG_INLINE
  const TranslationType& translation() const {
    return translation_;
  }

  EIGEN_STRONG_INLINE
  const SO3Type& so3() const {
    return so3_;
  }

protected:
  const TranslationType translation_;
  const SO3Type so3_;
};

}

#endif
