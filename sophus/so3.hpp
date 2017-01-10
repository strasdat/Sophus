// This file is part of Sophus.
//
// Copyright 2011-2013 Hauke Strasdat
// Copyrifht 2012-2013 Steven Lovegrove
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

#ifndef SOPHUS_SO3_HPP
#define SOPHUS_SO3_HPP

#include "common.hpp"

// Include only the selective set of Eigen headers that we need.
// This helps when using Sophus with unusual compilers, like nvcc.
#include <Eigen/src/Geometry/OrthoMethods.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/src/Geometry/RotationBase.h>

namespace Sophus {
template <typename _Scalar, int _Options = 0>
class SO3Group;
typedef SO3Group<double> SO3d;
typedef SO3Group<float> SO3f;
}  // namespace Sophus

namespace Eigen {
namespace internal {

template <typename _Scalar, int _Options>
struct traits<Sophus::SO3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Eigen::Quaternion<Scalar> QuaternionType;
};

template <typename _Scalar, int _Options>
struct traits<Map<Sophus::SO3Group<_Scalar>, _Options>>
    : traits<Sophus::SO3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<Eigen::Quaternion<Scalar>, _Options> QuaternionType;
};

template <typename _Scalar, int _Options>
struct traits<Map<const Sophus::SO3Group<_Scalar>, _Options>>
    : traits<const Sophus::SO3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<const Eigen::Quaternion<Scalar>, _Options> QuaternionType;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {
using std::sqrt;
using std::abs;
using std::cos;
using std::sin;

// SO3 base type - implements SO3 class but is storage agnostic.
//
// SO(3) is the group of rotations in 3d. As a matrix group, it is the set of
// matrices which are orthogonal such that ``R * R' = I`` (with ``R'`` being the
// transpose of ``R``) and have a positive determinant. In particular, the
// determinant is 1. Internally, the group is represented as a unit quaternion.
// Unit quaternion can be seen as members of the special unitary group SU(2).
// SU(2) is a double cover of SO(3). Hence, for every rotation matrix ``R``,
// there exists two unit quaternion: ``(r, v)`` and ``(r, -v)``, with ``r`` the
// real part and ``v`` being the imaginary 3-vector part of the quaternion.
//
// SO(3) is a compact, but non-commutative group. First it is compact since the
// set of rotation matrices is a closed and bounded set. Second it is
// non-commutative since the equation ``R_1 * R_2 = R_2 * R_1`` does not hold in
// general. For example rotating an object by some degrees about its ``x``-axis
// and then by some degrees about its y axis, does not lead to the same
// orienation when rotation first about ``y`` and then about ``x``.
//
// Class invairant: The 2-norm of ``unit_quaternion`` must be close to 1.
// Technically speaking, it must hold that:
//
//   ``|unit_quaternion().squaredNorm() - 1| <= Constants<Scalar>::epsilon()``.
template <typename Derived>
class SO3GroupBase {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using QuaternionReference =
      typename Eigen::internal::traits<Derived>::QuaternionType&;
  using ConstQuaternionReference =
      typename Eigen::internal::traits<Derived>::QuaternionType const&;

  // Degrees of freedom of group, number of dimensions in tangent space.
  static const int DoF = 3;
  // Number of internal parameters used (quaternion is a 4-tuple).
  static const int num_parameters = 4;
  // Group transformations are 3x3 matrices.
  static const int N = 3;
  typedef Eigen::Matrix<Scalar, N, N> Transformation;
  typedef Eigen::Matrix<Scalar, 3, 1> Point;
  typedef Eigen::Matrix<Scalar, DoF, 1> Tangent;
  typedef Eigen::Matrix<Scalar, DoF, DoF> Adjoint;

  // Adjoint transformation
  //
  // This function return the adjoint transformation ``Ad`` of the group
  // element ``A`` such that for all ``x`` it holds that
  // ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  //
  // For SO(3), it simply returns the rotation matrix corresponding to ``A``.
  //
  SOPHUS_FUNC Adjoint Adj() const { return matrix(); }

  // Returns copy of instance casted to NewScalarType.
  //
  template <typename NewScalarType>
  SOPHUS_FUNC SO3Group<NewScalarType> cast() const {
    return SO3Group<NewScalarType>(
        unit_quaternion().template cast<NewScalarType>());
  }

  // This provides unsafe read/write access to internal data. SO(3) is
  // represented by an Eigen::Quaternion (four parameters). When using direct
  // write access, the user needs to take care of that the quaternion stays
  // normalized.
  //
  // Note: The first three Scalars represent the imaginary parts, while the
  // forth Scalar represent the real part.
  //
  SOPHUS_FUNC Scalar* data() {
    return unit_quaternion_nonconst().coeffs().data();
  }

  // Const version of data() above.
  //
  SOPHUS_FUNC const Scalar* data() const {
    return unit_quaternion().coeffs().data();
  }

  // Returns ``*this`` times the ith generator of internal SU(2) representation.
  //
  SOPHUS_FUNC Eigen::Matrix<Scalar, num_parameters, 1>
  internalMultiplyByGenerator(int i) const {
    Eigen::Matrix<Scalar, num_parameters, 1> res;
    Eigen::Quaternion<Scalar> internal_gen_q;
    internalGenerator(i, &internal_gen_q);
    res.template head<4>() = (unit_quaternion() * internal_gen_q).coeffs();
    return res;
  }

  // Returns Jacobian of generator of internal SU(2) representation.
  //
  SOPHUS_FUNC
  Eigen::Matrix<Scalar, num_parameters, DoF> internalJacobian() const {
    Eigen::Matrix<Scalar, num_parameters, DoF> J;
    for (int i = 0; i < DoF; ++i) {
      J.col(i) = internalMultiplyByGenerator(i);
    }
    return J;
  }

  // Returns group inverse.
  //
  SOPHUS_FUNC SO3Group<Scalar> inverse() const {
    return SO3Group<Scalar>(unit_quaternion().conjugate());
  }

  // Logarithmic map
  //
  // Returns tangent space representation (= rotation vector) of the instance.
  //
  SOPHUS_FUNC Tangent log() const { return SO3Group<Scalar>::log(*this); }

  // It re-normalizes ``unit_quaternion`` to unit length.
  //
  // Note: Because of the class invariant, there is typically no need to call
  // this function directly.
  //
  SOPHUS_FUNC void normalize() {
    Scalar length = unit_quaternion_nonconst().norm();
    SOPHUS_ENSURE(length >= Constants<Scalar>::epsilon(),
                  "Quaternion (%) should not be close to zero!",
                  unit_quaternion_nonconst().coeffs().transpose());
    unit_quaternion_nonconst().coeffs() /= length;
  }

  // Returns 3x3 matrix representation of the instance.
  //
  // For SO(3), the matrix representation is an orthogonal matrix ``R`` with
  // ``det(R)=1``, thus the so-called "rotation matrix".
  //
  SOPHUS_FUNC Transformation matrix() const {
    return unit_quaternion().toRotationMatrix();
  }

  // Assignment operator.
  //
  template <typename OtherDerived>
  SOPHUS_FUNC SO3GroupBase<Derived>& operator=(
      const SO3GroupBase<OtherDerived>& other) {
    unit_quaternion_nonconst() = other.unit_quaternion();
    return *this;
  }

  // Group multiplication, which is rotation concatenation.
  //
  SOPHUS_FUNC SO3Group<Scalar> operator*(const SO3Group<Scalar>& other) const {
    SO3Group<Scalar> result(*this);
    result *= other;
    return result;
  }

  // Group action on 3-points.
  //
  // This function rotates a 3 dimensional point ``p`` by the SO3 element
  //  ``bar_R_foo`` (= rotation matrix): ``p_bar = bar_R_foo * p_foo``.
  //
  // Since SO3 is internally represented by a unit quaternion ``q``, it is
  // implemented as ``p_bar = q * p_foo * q^{*}``
  // with ``q^{*}`` being the quaternion conjugate of ``q``.
  //
  // Geometrically, ``p``  is rotated by angle ``|omega|`` around the
  // axis ``omega/|omega|`` with ``omega := vee(log(bar_R_foo))``.
  //
  // For ``vee``-operator, see below.
  //
  SOPHUS_FUNC Point operator*(const Point& p) const {
    return unit_quaternion()._transformVector(p);
  }

  // In-place group multiplication.
  //
  SOPHUS_FUNC SO3GroupBase<Derived>& operator*=(const SO3Group<Scalar>& other) {
    unit_quaternion_nonconst() *= other.unit_quaternion();

    Scalar squared_norm = unit_quaternion().squaredNorm();

    // We can assume that the squared-norm is close to 1 since we deal with a
    // unit quaternion. Due to numerical precision issues, there might
    // be a small drift after pose concatenation. Hence, we need to renormalizes
    // the quaternion here.
    // Since squared-norm is close to 1, we do not need to calculate the costly
    // square-root, but can use an approximation around 1 (see
    // http://stackoverflow.com/a/12934750 for details).
    if (squared_norm != Scalar(1.0)) {
      unit_quaternion_nonconst().coeffs() *=
          Scalar(2.0) / (Scalar(1.0) + squared_norm);
    }
    return *this;
  }

  // Takes in quaternion, and normalizes it.
  //
  // Precondition: The quaternion must not be close to zero.
  //
  SOPHUS_FUNC void setQuaternion(const Eigen::Quaternion<Scalar>& quaternion) {
    unit_quaternion_nonconst() = quaternion;
    normalize();
  }

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC ConstQuaternionReference unit_quaternion() const {
    return static_cast<const Derived*>(this)->unit_quaternion();
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  // Derivative of Lie bracket with respect to first element.
  //
  // This function returns ``D_a [a, b]`` with ``D_a`` being the
  // differential operator with respect to ``a``, ``[a, b]`` being the lie
  // bracket of the Lie algebra so3.
  // See ``lieBracket()`` below.
  //
  SOPHUS_FUNC static Adjoint d_lieBracketab_by_d_a(const Tangent& b) {
    return -hat(b);
  }

  // Group exponential
  //
  // This functions takes in an element of tangent space (= rotation vector
  // ``omega``) and returns the corresponding element of the group SO(3).
  //
  // To be more specific, this function computes ``expmat(hat(omega))``
  // with ``expmat(.)`` being the matrix exponential and ``hat(.)`` being the
  // hat()-operator of SO(3).
  //
  SOPHUS_FUNC static SO3Group<Scalar> exp(const Tangent& omega) {
    Scalar theta;
    return expAndTheta(omega, &theta);
  }

  // As above, but also returns ``theta = |omega|`` as out-parameter.
  //
  // Precondition: ``theta`` must not be ``nullptr``.
  //
  SOPHUS_FUNC static SO3Group<Scalar> expAndTheta(const Tangent& omega,
                                                  Scalar* theta) {
    Scalar theta_sq = omega.squaredNorm();
    *theta = sqrt(theta_sq);
    Scalar half_theta = static_cast<Scalar>(0.5) * (*theta);

    Scalar imag_factor;
    Scalar real_factor;
    if ((*theta) < Constants<Scalar>::epsilon()) {
      Scalar theta_po4 = theta_sq * theta_sq;
      imag_factor = static_cast<Scalar>(0.5) -
                    static_cast<Scalar>(1.0 / 48.0) * theta_sq +
                    static_cast<Scalar>(1.0 / 3840.0) * theta_po4;
      real_factor = static_cast<Scalar>(1) -
                    static_cast<Scalar>(0.5) * theta_sq +
                    static_cast<Scalar>(1.0 / 384.0) * theta_po4;
    } else {
      Scalar sin_half_theta = sin(half_theta);
      imag_factor = sin_half_theta / (*theta);
      real_factor = cos(half_theta);
    }

   SO3Group<Scalar> q;
   q.unit_quaternion_nonconst() = Eigen::Quaternion<Scalar>(
               real_factor, imag_factor * omega.x(), imag_factor * omega.y(),
               imag_factor * omega.z());
   SOPHUS_ENSURE(abs(q.unit_quaternion().squaredNorm() - Scalar(1)) < Sophus::Constants<Scalar>::epsilon(),
                 "SO3::exp failed! omega: %, real: %, img: %", omega.transpose(), real_factor, imag_factor);
    return q;
  }

  // Returns the ith infinitesimal generators of SO(3).
  //
  // The infinitesimal generators of SO(3) are:
  //
  //         |  0  0  0 |
  //   G_0 = |  0  0 -1 |
  //         |  0  1  0 |
  //
  //         |  0  0  1 |
  //   G_1 = |  0  0  0 |
  //         | -1  0  0 |
  //
  //         |  0 -1  0 |
  //   G_2 = |  1  0  0 |
  //         |  0  0  0 |
  //
  // Precondition: ``i`` must be 0, 1 or 2.
  //
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 && i <= 2, "i should be in range [0,2].");
    Tangent e;
    e.setZero();
    e[i] = static_cast<Scalar>(1);
    return hat(e);
  }

  // Returns the ith generator of internal SU(2) representation.
  //
  // Precondition: ``i`` must be 0, 1 or 2.
  //
  SOPHUS_FUNC static void internalGenerator(
      int i, Eigen::Quaternion<Scalar>* internal_gen_q) {
    SOPHUS_ENSURE(i >= 0 && i <= 2, "i should be in range [0,2]");
    SOPHUS_ENSURE(internal_gen_q != NULL,
                  "internal_gen_q must not be the null pointer");
    // Factor of 0.5 since SU(2) is a double cover of SO(3).
    internal_gen_q->coeffs()[i] = static_cast<Scalar>(0.5);
  }

  // hat-operator
  //
  // It takes in the 3-vector representation ``omega`` (= rotation vector) and
  // returns the corresponding matrix representation of Lie algebra element.
  //
  // Formally, the ``hat()`` operator of SO(3) is defined as
  //
  //   ``hat(.): R^3 -> R^{3x3},  hat(omega) = sum_i omega_i * G_i``
  //   (for i=0,1,2)
  //
  // with ``G_i`` being the ith infinitesimal generator of SO(3).
  //
  // The corresponding inverse is the ``vee``-operator, see below.
  //
  SOPHUS_FUNC static Transformation hat(const Tangent& omega) {
    Transformation Omega;
    // clang-format off
    Omega <<
        Scalar(0), -omega(2),  omega(1),
         omega(2), Scalar(0), -omega(0),
        -omega(1),  omega(0), Scalar(0);
    // clang-format on
    return Omega;
  }

  // Lie bracket
  //
  // It computes the Lie bracket of SO(3). To be more specific, it computes
  //
  //   ``[omega_1, omega_2]_so3 := vee([hat(omega_1), hat(omega_2)])``
  //
  // with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.) the
  // hat-operator and ``vee(.)`` the vee-operator of SO3.
  //
  // For the Lie algebra so3, the Lie bracket is simply the cross product:
  //
  // ``[omega_1, \omega_2]_so3 = omega_1 x \omega_2.``
  //
  SOPHUS_FUNC static Tangent lieBracket(const Tangent& omega1,
                                        const Tangent& omega2) {
    return omega1.cross(omega2);
  }

  // Logarithmic map
  //
  // Computes the logarithm, the inverse of the group exponential which maps
  // element of the group (rotation matrices) to elements of the tangent space
  // (rotation-vector).
  //
  // To be specific, this function computes ``vee(logmat(.))`` with
  // ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  // of SO(3).
  //
  SOPHUS_FUNC static Tangent log(const SO3Group<Scalar>& other) {
    Scalar theta;
    return logAndTheta(other, &theta);
  }

  // As above, but also returns ``theta = |omega|`` as out-parameter.
  //
  SOPHUS_FUNC static Tangent logAndTheta(const SO3Group<Scalar>& other,
                                         Scalar* theta) {
    Scalar squared_n = other.unit_quaternion().vec().squaredNorm();
    Scalar n = sqrt(squared_n);
    Scalar w = other.unit_quaternion().w();

    Scalar two_atan_nbyw_by_n;

    // Atan-based log thanks to
    //
    // C. Hertzberg et al.:
    // "Integrating Generic Sensor Fusion Algorithms with Sound State
    // Representation through Encapsulation of Manifolds"
    // Information Fusion, 2011

    if (n < Constants<Scalar>::epsilon()) {
      // If quaternion is normalized and n=0, then w should be 1;
      // w=0 should never happen here!
      SOPHUS_ENSURE(abs(w) >= Constants<Scalar>::epsilon(),
                    "Quaternion (%) should be normalized!",
                    other.unit_quaternion().coeffs().transpose());
      Scalar squared_w = w * w;
      two_atan_nbyw_by_n =
          static_cast<Scalar>(2) / w -
          static_cast<Scalar>(2) * (squared_n) / (w * squared_w);
    } else {
      if (abs(w) < Constants<Scalar>::epsilon()) {
        if (w > static_cast<Scalar>(0)) {
          two_atan_nbyw_by_n = Constants<Scalar>::pi() / n;
        } else {
          two_atan_nbyw_by_n = -Constants<Scalar>::pi() / n;
        }
      } else {
        two_atan_nbyw_by_n = static_cast<Scalar>(2) * atan(n / w) / n;
      }
    }

    *theta = two_atan_nbyw_by_n * n;

    return two_atan_nbyw_by_n * other.unit_quaternion().vec();
  }

  // vee-operator
  //
  // It takes the 3x3-matrix representation ``Omega`` and maps it to the
  // corresponding vector representation of Lie algebra.
  //
  // This is the inverse of the hat-operator, see above.
  //
  SOPHUS_FUNC static Tangent vee(const Transformation& Omega) {
    return static_cast<Scalar>(0.5) * Tangent(Omega(2, 1) - Omega(1, 2),
                                              Omega(0, 2) - Omega(2, 0),
                                              Omega(1, 0) - Omega(0, 1));
  }

 private:
  // Mutator of unit_quaternion is private to ensure class invariant. That is
  // the quaternion must stay close to unit length.
  //
  SOPHUS_FUNC QuaternionReference unit_quaternion_nonconst() {
    return static_cast<Derived*>(this)->unit_quaternion_nonconst();
  }
};

// SO3 default type - Constructors and default storage for SO3 Type.
template <typename _Scalar, int _Options>
class SO3Group : public SO3GroupBase<SO3Group<_Scalar, _Options>> {
  typedef SO3GroupBase<SO3Group<_Scalar, _Options>> Base;

 public:
  typedef typename Eigen::internal::traits<SO3Group<_Scalar, _Options>>::Scalar
      Scalar;
  typedef typename Eigen::internal::traits<
      SO3Group<_Scalar, _Options>>::QuaternionType& QuaternionReference;
  typedef const typename Eigen::internal::traits<
      SO3Group<_Scalar, _Options>>::QuaternionType& ConstQuaternionReference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  // ``Base`` is friend so unit_quaternion_nonconst can be accessed from
  // ``Base``.
  friend class SO3GroupBase<SO3Group<_Scalar, _Options>>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor initialize unit quaternion to identity rotation.
  //
  SOPHUS_FUNC SO3Group()
      : unit_quaternion_(static_cast<Scalar>(1), static_cast<Scalar>(0),
                         static_cast<Scalar>(0), static_cast<Scalar>(0)) {}

  // Copy constructor
  //
  template <typename OtherDerived>
  SOPHUS_FUNC SO3Group(const SO3GroupBase<OtherDerived>& other)
      : unit_quaternion_(other.unit_quaternion()) {}

  // Constructor from rotation matrix
  //
  // Precondition: rotation matrix needs to be orthogonal with determinant of 1.
  //
  SOPHUS_FUNC SO3Group(const Transformation& R) : unit_quaternion_(R) {}

  // Constructor from quaternion
  //
  // Precondition: quaternion must not be close to zero.
  //
  SOPHUS_FUNC explicit SO3Group(const Eigen::Quaternion<Scalar>& quat)
      : unit_quaternion_(quat) {
    Base::normalize();
  }

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC ConstQuaternionReference unit_quaternion() const {
    return unit_quaternion_;
  }

 protected:
  // Mutator of unit_quaternion is protected to ensure class invariant.
  //
  SOPHUS_FUNC QuaternionReference unit_quaternion_nonconst() {
    return unit_quaternion_;
  }

  Eigen::Quaternion<Scalar> unit_quaternion_;
};

}  // namespace Sophus

namespace Eigen {

// Specialization of Eigen::Map for SO3GroupBase
//
// Allows us to wrap SO3 objects around POD array (e.g. external c style
// quaternion).
template <typename _Scalar, int _Options>
class Map<Sophus::SO3Group<_Scalar>, _Options>
    : public Sophus::SO3GroupBase<Map<Sophus::SO3Group<_Scalar>, _Options>> {
  typedef Sophus::SO3GroupBase<Map<Sophus::SO3Group<_Scalar>, _Options>> Base;

 public:
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  typedef typename Eigen::internal::traits<Map>::QuaternionType&
      QuaternionReference;
  typedef const typename Eigen::internal::traits<Map>::QuaternionType&
      ConstQuaternionReference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  // ``Base`` is friend so unit_quaternion_nonconst can be accessed from
  // ``Base``.
  friend class Sophus::SO3GroupBase<Map<Sophus::SO3Group<_Scalar>, _Options>>;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar* coeffs) : unit_quaternion_(coeffs) {}

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC ConstQuaternionReference unit_quaternion() const {
    return unit_quaternion_;
  }

 protected:
  // Mutator of unit_quaternion is protected to ensure class invariant.
  //
  SOPHUS_FUNC QuaternionReference unit_quaternion_nonconst() {
    return unit_quaternion_;
  }

  Map<Eigen::Quaternion<Scalar>, _Options> unit_quaternion_;
};

// Specialization of Eigen::Map for ``const SO3GroupBase``
//
// Allows us to wrap SO3 objects around POD array (e.g. external c style
// quaternion).
template <typename _Scalar, int _Options>
class Map<const Sophus::SO3Group<_Scalar>, _Options>
    : public Sophus::SO3GroupBase<
          Map<const Sophus::SO3Group<_Scalar>, _Options>> {
  typedef Sophus::SO3GroupBase<Map<const Sophus::SO3Group<_Scalar>, _Options>>
      Base;

 public:
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  typedef const typename Eigen::internal::traits<Map>::QuaternionType&
      ConstQuaternionReference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(const Scalar* coeffs) : unit_quaternion_(coeffs) {}

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC ConstQuaternionReference unit_quaternion() const {
    return unit_quaternion_;
  }

 protected:
  // Mutator of unit_quaternion is protected to ensure class invariant.
  //
  const Map<const Eigen::Quaternion<Scalar>, _Options> unit_quaternion_;
};
}

#endif
