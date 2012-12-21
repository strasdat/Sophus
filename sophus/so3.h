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

#ifndef SOPHUS_SO3_H
#define SOPHUS_SO3_H

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Geometry>

////////////////////////////////////////////////////////////////////////////
// Forward Declarations / typedefs
////////////////////////////////////////////////////////////////////////////

namespace Sophus {
template<typename _Scalar, int _Options=0> class SO3Group;
typedef SO3Group<double> SO3; //deprecated
typedef SO3Group<double> SO3d;
typedef SO3Group<float> SO3f;
}

////////////////////////////////////////////////////////////////////////////
// Eigen Traits (For querying derived types in CRTP hierarchy)
////////////////////////////////////////////////////////////////////////////

namespace Eigen {
namespace internal {

template<typename _Scalar, int _Options>
struct traits<Sophus::SO3Group<_Scalar,_Options> > {
  typedef _Scalar Scalar;
  typedef Quaternion<Scalar> QuaternionType;
};

template<typename _Scalar, int _Options>
struct traits<Map<Sophus::SO3Group<_Scalar>, _Options> >
    : traits<Sophus::SO3Group<_Scalar, _Options> > {
  typedef _Scalar Scalar;
  typedef Map<Quaternion<Scalar>,_Options> QuaternionType;
};

template<typename _Scalar, int _Options>
struct traits<Map<const Sophus::SO3Group<_Scalar>, _Options> >
    : traits<const Sophus::SO3Group<_Scalar, _Options> > {
  typedef _Scalar Scalar;
  typedef Map<const Quaternion<Scalar>,_Options> QuaternionType;
};

}
}

namespace Sophus {
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////
// SO3GroupBase type - implements SO3 class but is storage agnostic
////////////////////////////////////////////////////////////////////////////

template<typename Scalar>
struct SophusConstants {
  EIGEN_ALWAYS_INLINE static
  const Scalar epsilon() {
    return static_cast<Scalar>(1e-10);
  }

  EIGEN_ALWAYS_INLINE static
  const Scalar pi() {
    return static_cast<Scalar>(M_PI);
  }
};

template<>
struct SophusConstants<float> {
  EIGEN_ALWAYS_INLINE static
  const float epsilon() {
    return 1e-5f;
  }

  EIGEN_ALWAYS_INLINE static
  const float pi() {
    return static_cast<float>(M_PI);
  }
};

template<typename Derived>
class SO3GroupBase {
public:
  typedef typename internal::traits<Derived>::Scalar Scalar;
  typedef typename internal::traits<Derived>::QuaternionType QuaternionType;
  static const int DoF = 3;

  inline
  SO3Group<Scalar>& operator=(const SO3Group<Scalar> & other) {
    unit_quaternion() = other.unit_quaternion();
    return *this;
  }

  inline
  const SO3Group<Scalar> operator*(const SO3Group<Scalar>& other) const {
    SO3Group<Scalar> result(*this);
    result.unit_quaternion() *= other.unit_quaternion();
    result.unit_quaternion().normalize();
    return result;
  }

  inline
  void operator*=(const SO3Group<Scalar>& other) {
    unit_quaternion() *= other.unit_quaternion();
    unit_quaternion().normalize();
  }

  inline
  const Matrix<Scalar,3,1> operator*(const Matrix<Scalar,3,1> & xyz) const {
    return unit_quaternion()._transformVector(xyz);
  }

  inline
  const SO3Group<Scalar> inverse() const {
    return SO3Group<Scalar>(unit_quaternion().conjugate());
  }

  inline
  const Matrix<Scalar,3,3> matrix() const {
    return unit_quaternion().toRotationMatrix();
  }

  inline
  const Matrix<Scalar,3,3> Adj() const {
    return matrix();
  }

  inline
  const Matrix<Scalar,3,3> generator(int i) {
    assert(i>=0 && i<3);
    Matrix<Scalar,3,1> e;
    e.setZero();
    e[i] = 1.f;
    return hat(e);
  }

  inline
  const Matrix<Scalar,3,1> log() const {
    return SO3Group<Scalar>::log(*this);
  }

  inline static
  const Matrix<Scalar,3,1> log(const SO3Group<Scalar> & other) {
    Scalar theta;
    return logAndTheta(other, &theta);
  }

  inline static
  const Matrix<Scalar,3,1> logAndTheta(const SO3Group<Scalar> & other,
                                 Scalar * theta) {
    const Scalar squared_n = other.unit_quaternion().vec().squaredNorm();
    const Scalar n = sqrt(squared_n);
    const Scalar w = other.unit_quaternion().w();

    Scalar two_atan_nbyw_by_n;

    // Atan-based log thanks to
    //
    // C. Hertzberg et al.:
    // "Integrating Generic Sensor Fusion Algorithms with Sound State
    // Representation through Encapsulation of Manifolds"
    // Information Fusion, 2011

    if (n < SophusConstants<Scalar>::epsilon()) {
      // If quaternion is normalized and n=1, then w should be 1;
      // w=0 should never happen here!
      assert(fabs(w)>SophusConstants<Scalar>::epsilon());
      const Scalar squared_w = w*w;
      two_atan_nbyw_by_n = static_cast<Scalar>(2) / w
                           - static_cast<Scalar>(2)*(squared_n)/(w*squared_w);
    } else {
      if (fabs(w)<SophusConstants<Scalar>::epsilon()) {
        if (w > static_cast<Scalar>(0)) {
          two_atan_nbyw_by_n = M_PI/n;
        } else {
          two_atan_nbyw_by_n = -M_PI/n;
        }
      }else{
        two_atan_nbyw_by_n = static_cast<Scalar>(2) * atan(n/w) / n;
      }
    }

    *theta = two_atan_nbyw_by_n*n;

    return two_atan_nbyw_by_n * other.unit_quaternion().vec();
  }

  inline static
  const SO3Group<Scalar> exp(const Matrix<Scalar,3,1> & omega) {
    Scalar theta;
    return expAndTheta(omega, &theta);
  }

  inline static
  const SO3Group<Scalar> expAndTheta(const Matrix<Scalar,3,1> & omega,
                               Scalar * theta) {
    const Scalar theta_sq = omega.squaredNorm();
    *theta = sqrt(theta_sq);
    const Scalar half_theta = 0.5*(*theta);

    Scalar imag_factor;
    Scalar real_factor;;
    if((*theta)<SophusConstants<Scalar>::epsilon()) {
      const Scalar theta_po4 = theta_sq*theta_sq;
      imag_factor = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_po4;
      real_factor = static_cast<Scalar>(1)
                    - static_cast<Scalar>(0.5)*theta_sq +
                    static_cast<Scalar>(1.0/384.0)*theta_po4;
    } else {
      const Scalar sin_half_theta = sin(half_theta);
      imag_factor = sin_half_theta/(*theta);
      real_factor = cos(half_theta);
    }

    return SO3Group<Scalar>(QuaternionType(real_factor,
                                           imag_factor*omega.x(),
                                           imag_factor*omega.y(),
                                           imag_factor*omega.z()));
  }

  inline static
  const Matrix<Scalar,3,3> hat(const Matrix<Scalar,3,1> & v) {
    Matrix<Scalar,3,3> Omega;
    Omega <<  static_cast<Scalar>(0), -v(2),  v(1)
        ,  v(2),     static_cast<Scalar>(0), -v(0)
        , -v(1),  v(0),     static_cast<Scalar>(0);
    return Omega;
  }

  inline static
  const Matrix<Scalar,3,1> vee(const Matrix<Scalar,3,3> & Omega) {
    assert(fabs(Omega(2,1)+Omega(1,2)) < SophusConstants<Scalar>::epsilon());
    assert(fabs(Omega(0,2)+Omega(2,0)) < SophusConstants<Scalar>::epsilon());
    assert(fabs(Omega(1,0)+Omega(0,1)) < SophusConstants<Scalar>::epsilon());
    return Matrix<Scalar,3,1>(Omega(2,1), Omega(0,2), Omega(1,0));
  }

  inline static
  const Matrix<Scalar,3,1> lieBracket(const Matrix<Scalar,3,1> & omega1,
                                const Matrix<Scalar,3,1> & omega2) {
    return omega1.cross(omega2);
  }

  inline static
  const Matrix<Scalar,3,3> d_lieBracketab_by_d_a(const Matrix<Scalar,3,1> & b) {
    return -hat(b);
  }

  // GETTERS & SETTERS

  EIGEN_STRONG_INLINE
  QuaternionType& unit_quaternion() {
    return static_cast<Derived*>(this)->unit_quaternion();
  }

  EIGEN_STRONG_INLINE
  const QuaternionType& unit_quaternion() const {
    return static_cast<const Derived*>(this)->unit_quaternion();
  }

  inline
  void setQuaternion(const QuaternionType& quaternion) {
    assert(quaternion.norm()!=static_cast<Scalar>(0));
    unit_quaternion() = quaternion;
    unit_quaternion().normalize();
  }

  template<typename NewScalarType>
  inline SO3Group<NewScalarType> cast() const {
    return SO3Group<NewScalarType>(unit_quaternion()
                                   .template cast<NewScalarType>() );
  }

  inline Scalar* data() {
    return unit_quaternion().data();
  }

  inline const Scalar* data() const {
    return unit_quaternion().data();
  }

};

////////////////////////////////////////////////////////////////////////////
// SO3Group type - Constructors and default storage for SO3 Type
////////////////////////////////////////////////////////////////////////////

template<typename _Scalar, int _Options>
class SO3Group : public SO3GroupBase<SO3Group<_Scalar,_Options> >
{
public:
  typedef typename internal::traits<SO3Group<_Scalar,_Options> >
  ::Scalar Scalar;
  typedef typename internal::traits<SO3Group<_Scalar,_Options> >
  ::QuaternionType QuaternionType;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline SO3Group()
    // Initialize Quaternion to identity rotation
    : unit_quaternion_(static_cast<Scalar>(0), static_cast<Scalar>(0),
                       static_cast<Scalar>(0), static_cast<Scalar>(1)) {
  }

  template<typename OtherDerived> inline
  SO3Group(const SO3GroupBase<OtherDerived> & other)
    : unit_quaternion_(other.unit_quaternion()) {
  }


  inline SO3Group(const Matrix<Scalar,3,3> & R) : unit_quaternion_(R) {
  }

  inline SO3Group(const QuaternionType & quat) : unit_quaternion_(quat) {
    assert(unit_quaternion_.squaredNorm() > SophusConstants<Scalar>::epsilon());
    unit_quaternion_.normalize();
  }

  inline SO3Group(Scalar rot_x, Scalar rot_y, Scalar rot_z) {
    unit_quaternion_
        = (SO3Group::exp(Matrix<Scalar,3,1>(rot_x, 0.f, 0.f))
           *SO3Group::exp(Matrix<Scalar,3,1>(0.f, rot_y, 0.f))
           *SO3Group::exp(Matrix<Scalar,3,1>(0.f, 0.f, rot_z)))
          .unit_quaternion_;
  }

  // GETTERS & SETTERS

  EIGEN_STRONG_INLINE
  const QuaternionType & unit_quaternion() const {
    return unit_quaternion_;
  }

  EIGEN_STRONG_INLINE
  QuaternionType & unit_quaternion() {
    return unit_quaternion_;
  }

protected:
  QuaternionType unit_quaternion_;
};

} // end namespace

////////////////////////////////////////////////////////////////////////////
// Specialisation of Eigen::Map for SO3GroupBase
// Allows us to wrap SO3 Objects around POD array
// (e.g. external c style quaternion)
////////////////////////////////////////////////////////////////////////////

namespace Eigen {

template<typename _Scalar, int _Options>
class Map<Sophus::SO3Group<_Scalar>, _Options>
    : public Sophus::SO3GroupBase<Map<Sophus::SO3Group<_Scalar>, _Options> > {
  typedef Sophus::SO3GroupBase<Map<Sophus::SO3Group<_Scalar>, _Options> > Base;

public:
  typedef typename internal::traits<Map>::Scalar Scalar;
  typedef typename internal::traits<Map>::QuaternionType QuaternionType;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  EIGEN_STRONG_INLINE
  Map(Scalar* coeffs) : unit_quaternion_(coeffs) {
  }

  // GETTERS & SETTERS

  EIGEN_STRONG_INLINE
  const QuaternionType & unit_quaternion() const {
    return unit_quaternion_;
  }

  EIGEN_STRONG_INLINE
  QuaternionType & unit_quaternion() {
    return unit_quaternion_;
  }

protected:
  QuaternionType unit_quaternion_;
};

template<typename _Scalar, int _Options>
class Map<const Sophus::SO3Group<_Scalar>, _Options>
    : public Sophus::SO3GroupBase<
    Map<const Sophus::SO3Group<_Scalar>, _Options> > {
  typedef Sophus::SO3GroupBase<Map<const Sophus::SO3Group<_Scalar>, _Options> >
  Base;

public:
  typedef typename internal::traits<Map>::Scalar Scalar;
  typedef typename internal::traits<Map>::QuaternionType QuaternionType;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  // GETTERS & SETTERS

  EIGEN_STRONG_INLINE
  Map(const Scalar* coeffs) : unit_quaternion_(coeffs) {
  }

  EIGEN_STRONG_INLINE
  const QuaternionType & unit_quaternion() const {
    return unit_quaternion_;
  }

protected:
  const QuaternionType unit_quaternion_;
};

}

#endif
