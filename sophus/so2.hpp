// This file is part of Sophus.
//
// Copyright 2012-2013 Hauke Strasdat
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

#ifndef SOPHUS_SO2_HPP
#define SOPHUS_SO2_HPP

#include <complex>

// Include only the selective set of Eigen headers that we need.
// This helps when using Sophus with unusual compilers, like nvcc.
#include <Eigen/LU>

#include "common.hpp"

namespace Sophus {
template <typename _Scalar, int _Options = 0>
class SO2Group;
typedef SO2Group<double> SO2d; /**< double precision SO2 */
typedef SO2Group<float> SO2f;  /**< single precision SO2 */
}  // namespace Sophus

namespace Eigen {
namespace internal {

template <typename _Scalar, int _Options>
struct traits<Sophus::SO2Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Matrix<Scalar, 2, 1> ComplexType;
};

template <typename _Scalar, int _Options>
struct traits<Map<Sophus::SO2Group<_Scalar>, _Options>>
    : traits<Sophus::SO2Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<Matrix<Scalar, 2, 1>, _Options> ComplexType;
};

template <typename _Scalar, int _Options>
struct traits<Map<const Sophus::SO2Group<_Scalar>, _Options>>
    : traits<const Sophus::SO2Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<const Matrix<Scalar, 2, 1>, _Options> ComplexType;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {

// SO2 base type - implements SO2 class but is storage agnostic.
//
// SO(2) is the group of rotations in 2d. As a matrix group, it is the set of
// matrices which are orthogonal such that ``R * R' = I`` (with ``R'`` being the
// transpose of ``R``) and have a positive determinant. In particular, the
// determinant is 1. Let ``theta`` be the rotation angle, the rotation matrix
// can be written in close form:
//
//  | cos(theta) -sin(theta) |
//  | sin(theta)  cos(theta) |
//
// As a matter of fact, the first column of those matrices is isomorph to the
// set of unit complex numbers U(1). Thus, internally, SO2 is represented as
// complex number with length 1.
//
// SO(2) is a compact and commutative group. First it is compact since the set
// of rotation matrices is a closed and bounded set. Second it is commutative
// since ``R(alpha) * R(beta) = R(beta) * R(alpha``,  simply because ``alpha +
// beta = beta + alpha`` with ``alpha`` and ``beta`` being rotation angles
// (about the same axis).
//
// Class invairant: The 2-norm of ``unit_complex`` must be close to 1.
// Technically speaking, it must hold that:
//
//   ``|unit_complex().squaredNorm() - 1| <= Constants<Scalar>::epsilon()``.
template <typename Derived>
class SO2GroupBase {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using ComplexReference =
      typename Eigen::internal::traits<Derived>::ComplexType&;
  using ConstComplexReference =
      const typename Eigen::internal::traits<Derived>::ComplexType&;

  // Degrees of freedom of manifold, number of dimensions in tangent space (one
  // since we only have in-plane rotations).
  static const int DoF = 1;
  // Number of internal parameters used (complex numbers are a tuples).
  static const int num_parameters = 2;
  // Group transformations are 2x2 matrices.
  static const int N = 2;
  using Transformation = Eigen::Matrix<Scalar, N, N>;
  using Point = Eigen::Matrix<Scalar, 2, 1>;
  using Tangent = Scalar;
  using Adjoint = Scalar;

  // Adjoint transformation
  //
  // This function return the adjoint transformation ``Ad`` of the group
  // element ``A`` such that for all ``x`` it holds that
  // ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  //
  // It simply ``1``, since ``SO(2)`` is a commutative group.
  //
  SOPHUS_FUNC Adjoint Adj() const { return 1; }

  // Returns copy of instance casted to NewScalarType.
  //
  template <typename NewScalarType>
  SOPHUS_FUNC SO2Group<NewScalarType> cast() const {
    return SO2Group<NewScalarType>(
        unit_complex().template cast<NewScalarType>());
  }

  // This provides unsafe read/write access to internal data. SO(2) is
  // represented by a unit complex number (two parameters). When using direct
  // write access, the user needs to take care of that the complex number stays
  // normalized.
  //
  SOPHUS_FUNC Scalar* data() { return unit_complex_nonconst().data(); }

  // Const version of data() above.
  //
  SOPHUS_FUNC const Scalar* data() const { return unit_complex().data(); }

  // Returns ``*this`` times the ith generator of internal U(1) representation.
  //
  SOPHUS_FUNC SO2Group<Scalar> inverse() const {
    return SO2Group<Scalar>(unit_complex().x(), -unit_complex().y());
  }

  // Logarithmic map
  //
  // Returns tangent space representation (= rotation angle) of the instance.
  //
  SOPHUS_FUNC Scalar log() const { return SO2Group<Scalar>::log(*this); }

  // It re-normalizes ``unit_complex`` to unit length.
  //
  // Note: Because of the class invariant, there is typically no need to call
  // this function directly.
  //
  SOPHUS_FUNC void normalize() {
    Scalar length = std::sqrt(unit_complex().x() * unit_complex().x() +
                              unit_complex().y() * unit_complex().y());
    SOPHUS_ENSURE(length >= Constants<Scalar>::epsilon(),
                  "Complex number should not be close to zero!");
    unit_complex_nonconst().x() /= length;
    unit_complex_nonconst().y() /= length;
  }

  // Returns 2x2 matrix representation of the instance.
  //
  // For SO(2), the matrix representation is an orthogonal matrix ``R`` with
  // ``det(R)=1``, thus the so-called "rotation matrix".
  //
  SOPHUS_FUNC Transformation matrix() const {
    const Scalar& real = unit_complex().x();
    const Scalar& imag = unit_complex().y();
    Transformation R;
    // clang-format off
    R <<
        real, -imag,
        imag,  real;
    // clang-format on
    return R;
  }

  // Assignment operator
  //
  template <typename OtherDerived>
  SOPHUS_FUNC SO2GroupBase<Derived>& operator=(
      const SO2GroupBase<OtherDerived>& other) {
    unit_complex_nonconst() = other.unit_complex();
    return *this;
  }

  // Group multiplication, which is rotation concatenation.
  //
  SOPHUS_FUNC SO2Group<Scalar> operator*(const SO2Group<Scalar>& other) const {
    SO2Group<Scalar> result(*this);
    result *= other;
    return result;
  }

  // Group action on 3-points.
  //
  // This function rotates a 3 dimensional point ``p`` by the SO3 element
  //  ``bar_R_foo`` (= rotation matrix): ``p_bar = bar_R_foo * p_foo``.
  //
  SOPHUS_FUNC Point operator*(const Point& p) const {
    const Scalar& real = unit_complex().x();
    const Scalar& imag = unit_complex().y();
    return Point(real * p[0] - imag * p[1], imag * p[0] + real * p[1]);
  }

  // In-place group multiplication.
  //
  SOPHUS_FUNC SO2GroupBase<Derived> operator*=(const SO2Group<Scalar>& other) {
    Scalar lhs_real = unit_complex().x();
    Scalar lhs_imag = unit_complex().y();
    const Scalar& rhs_real = other.unit_complex().x();
    const Scalar& rhs_imag = other.unit_complex().y();
    // complex multiplication
    unit_complex_nonconst().x() = lhs_real * rhs_real - lhs_imag * rhs_imag;
    unit_complex_nonconst().y() = lhs_real * rhs_imag + lhs_imag * rhs_real;

    Scalar squared_norm = unit_complex_nonconst().squaredNorm();
    // We can assume that the squared-norm is close to 1 since we deal with a
    // unit complex number. Due to numerical precision issues, there might
    // be a small drift after pose concatenation. Hence, we need to renormalizes
    // the complex number here.
    // Since squared-norm is close to 1, we do not need to calculate the costly
    // square-root, but can use an approximation around 1 (see
    // http://stackoverflow.com/a/12934750 for details).
    if (squared_norm != Scalar(1.0)) {
      unit_complex_nonconst() *= Scalar(2.0) / (Scalar(1.0) + squared_norm);
    }
    return *this;
  }

  // Takes in complex number / tuple and normalizes it.
  //
  // Precondition: The complex number must not be close to zero.
  //
  SOPHUS_FUNC void setComplex(const Point& complex) {
    unit_complex_nonconst() = complex;
    normalize();
  }

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC
  ConstComplexReference unit_complex() const {
    return static_cast<const Derived*>(this)->unit_complex();
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  // Group exponential
  //
  // This functions takes in an element of tangent space (= rotation angle
  // ``theta``) and returns the corresponding element of the group SO(2).
  //
  // To be more specific, this function computes ``expmat(hat(omega))``
  // with ``expmat(.)`` being the matrix exponential and ``hat(.)`` being the
  // hat()-operator of SO(2).
  //
  SOPHUS_FUNC static SO2Group<Scalar> exp(const Tangent& theta) {
    return SO2Group<Scalar>(std::cos(theta), std::sin(theta));
  }

  // Returns the infinitesimal generators of SO3.
  //
  // The infinitesimal generators of SO(2) is:
  //
  //   |  0  1 |
  //   | -1  0 |
  //
  SOPHUS_FUNC static Transformation generator() { return hat(1); }

  // hat-operator
  //
  // It takes in the scalar representation ``theta`` (= rotation angle) and
  // returns the corresponding matrix representation of Lie algebra element.
  //
  // Formally, the ``hat()`` operator of SO(2) is defined as
  //
  //   ``hat(.): R^2 -> R^{2x2},  hat(theta) = theta * G``
  //
  // with ``G`` being the infinitesimal generator of SO(2).
  //
  // The corresponding inverse is the ``vee``-operator, see below.
  //
  SOPHUS_FUNC static Transformation hat(const Tangent& theta) {
    Transformation Omega;
    Omega << static_cast<Scalar>(0), -theta, theta, static_cast<Scalar>(0);
    return Omega;
  }

  // Lie bracket
  //
  // It returns the Lie bracket of SO(2). Since SO(2) is a commutative group,
  // the Lie bracket is simple ``0``.
  //
  SOPHUS_FUNC static Tangent lieBracket(const Tangent&, const Tangent&) {
    return static_cast<Scalar>(0);
  }

  // Logarithmic map
  //
  // Computes the logarithm, the inverse of the group exponential which maps
  // element of the group (rotation matrices) to elements of the tangent space
  // (rotation angles).
  //
  // To be specific, this function computes ``vee(logmat(.))`` with
  // ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  // of SO(2).
  //
  SOPHUS_FUNC static Tangent log(const SO2Group<Scalar>& other) {
    using std::atan2;
    return atan2(other.unit_complex_.y(), other.unit_complex().x());
  }

  // vee-operator
  //
  // It takes the 2x2-matrix representation ``Omega`` and maps it to the
  // corresponding scalar representation of Lie algebra.
  //
  // This is the inverse of the hat-operator, see above.
  //
  SOPHUS_FUNC static Tangent vee(const Transformation& Omega) {
    return static_cast<Scalar>(0.5) * (Omega(1, 0) - Omega(0, 1));
  }

 private:
  // Mutator of unit_complex is private to ensure class invariant. That is
  // the complex number must stay close to unit length.
  //
  SOPHUS_FUNC
  ComplexReference unit_complex_nonconst() {
    return static_cast<Derived*>(this)->unit_complex_nonconst();
  }
};

// SO2 default type - Constructors and default storage for SO2 Type
template <typename _Scalar, int _Options>
class SO2Group : public SO2GroupBase<SO2Group<_Scalar, _Options>> {
  typedef SO2GroupBase<SO2Group<_Scalar, _Options>> Base;

 public:
  using Scalar =
      typename Eigen::internal::traits<SO2Group<_Scalar, _Options>>::Scalar;
  using ComplexReference = typename Eigen::internal::traits<
      SO2Group<_Scalar, _Options>>::ComplexType&;
  using ConstComplexReference = const typename Eigen::internal::traits<
      SO2Group<_Scalar, _Options>>::ComplexType&;

  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  // ``Base`` is friend so unit_complex_nonconst can be accessed from ``Base``.
  friend class SO2GroupBase<SO2Group<_Scalar, _Options>>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor initialize unit complex number to identity rotation.
  //
  SOPHUS_FUNC SO2Group()
      : unit_complex_(static_cast<Scalar>(1), static_cast<Scalar>(0)) {}

  // Copy constructor
  //
  template <typename OtherDerived>
  SOPHUS_FUNC SO2Group(const SO2GroupBase<OtherDerived>& other)
      : unit_complex_(other.unit_complex()) {}

  // Constructor from rotation matrix
  //
  // Precondition: rotation matrix need to be orthogonal with determinant of 1.
  //
  SOPHUS_FUNC explicit SO2Group(const Transformation& R)
      : unit_complex_(static_cast<Scalar>(0.5) * (R(0, 0) + R(1, 1)),
                      static_cast<Scalar>(0.5) * (R(1, 0) - R(0, 1))) {
    SOPHUS_ENSURE(std::abs(R.determinant() - static_cast<Scalar>(1)) <=
                      Constants<Scalar>::epsilon(),
                  "det(R) should be (close to) 1.");
  }

  // Constructor from pair of real and imaginary number.
  //
  // Precondition: The pair must not be close to zero.
  //
  SOPHUS_FUNC SO2Group(const Scalar& real, const Scalar& imag)
      : unit_complex_(real, imag) {
    Base::normalize();
  }

  // Constructor from 2-vector.
  //
  // Precondition: The vector must not be close to zero.
  //
  SOPHUS_FUNC explicit SO2Group(const Eigen::Matrix<Scalar, 2, 1>& complex)
      : unit_complex_(complex) {
    Base::normalize();
  }

  // Constructor from std::complex
  //
  // Precondition: ``complex`` number must not be zero
  //
  SOPHUS_FUNC explicit SO2Group(const std::complex<Scalar>& complex)
      : unit_complex_(complex.real(), complex.imag()) {
    Base::normalize();
  }

  // Constructor from an rotation angle.
  //
  SOPHUS_FUNC explicit SO2Group(Scalar theta) {
    unit_complex_nonconst() = SO2Group<Scalar>::exp(theta).unit_complex();
  }

  // Accessor of unit complex number
  //
  SOPHUS_FUNC ConstComplexReference unit_complex() const {
    return unit_complex_;
  }

 protected:
  // Mutator of complex number is protected to ensure class invariant.
  //
  SOPHUS_FUNC ComplexReference unit_complex_nonconst() { return unit_complex_; }

  static bool isNearZero(const Scalar& real, const Scalar& imag) {
    return (real * real + imag * imag < Constants<Scalar>::epsilon());
  }

  Eigen::Matrix<Scalar, 2, 1> unit_complex_;
};

}  // namespace Sophus

namespace Eigen {

// Specialization of Eigen::Map for ``SO2GroupBase``
//
// Allows us to wrap SO2 objects around POD array (e.g. external c style
// complex number / tuple).
template <typename _Scalar, int _Options>
class Map<Sophus::SO2Group<_Scalar>, _Options>
    : public Sophus::SO2GroupBase<Map<Sophus::SO2Group<_Scalar>, _Options>> {
  typedef Sophus::SO2GroupBase<Map<Sophus::SO2Group<_Scalar>, _Options>> Base;

 public:
  using Scalar = typename Eigen::internal::traits<Map>::Scalar;
  using ComplexReference = typename Eigen::internal::traits<Map>::ComplexType&;
  using ConstComplexReference =
      const typename Eigen::internal::traits<Map>::ComplexType&;

  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  // ``Base`` is friend so unit_complex_nonconst can be accessed from ``Base``.
  friend class Sophus::SO2GroupBase<Map<Sophus::SO2Group<_Scalar>, _Options>>;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  Map(Scalar* coeffs) : unit_complex_(coeffs) {}

  // Accessor of unit complex number.
  //
  SOPHUS_FUNC
  ConstComplexReference unit_complex() const { return unit_complex_; }

 protected:
  // Mutator of unit_complex is protected to ensure class invariant.
  //
  SOPHUS_FUNC
  ComplexReference unit_complex_nonconst() { return unit_complex_; }

  Map<Matrix<Scalar, 2, 1>, _Options> unit_complex_;
};

// Specialization of Eigen::Map for ``const SO2GroupBase``
//
// Allows us to wrap SO2 objects around POD array (e.g. external c style
// complex number / tuple).
template <typename _Scalar, int _Options>
class Map<const Sophus::SO2Group<_Scalar>, _Options>
    : public Sophus::SO2GroupBase<
          Map<const Sophus::SO2Group<_Scalar>, _Options>> {
  typedef Sophus::SO2GroupBase<Map<const Sophus::SO2Group<_Scalar>, _Options>>
      Base;

 public:
  typedef typename internal::traits<Map>::Scalar Scalar;
  typedef const typename internal::traits<Map>::ComplexType&
      ConstComplexReference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(const Scalar* coeffs) : unit_complex_(coeffs) {}

  // Accessor of unit complex number.
  //
  SOPHUS_FUNC ConstComplexReference unit_complex() const {
    return unit_complex_;
  }

 protected:
  // Mutator of unit_complex is protected to ensure class invariant.
  //
  const Map<const Matrix<Scalar, 2, 1>, _Options> unit_complex_;
};
}

#endif  // SOPHUS_SO2_HPP
