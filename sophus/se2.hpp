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

#ifndef SOPHUS_SE2_HPP
#define SOPHUS_SE2_HPP

#include "so2.hpp"

namespace Sophus {
template <typename _Scalar, int _Options = 0>
class SE2Group;
typedef SE2Group<double> SE2d; /**< double precision SE2 */
typedef SE2Group<float> SE2f;  /**< single precision SE2 */
}  // namespace Sophus

namespace Eigen {
namespace internal {

template <typename _Scalar, int _Options>
struct traits<Sophus::SE2Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Eigen::Matrix<Scalar, 2, 1> TranslationType;
  typedef Sophus::SO2Group<Scalar> SO2Type;
};

template <typename _Scalar, int _Options>
struct traits<Map<Sophus::SE2Group<_Scalar>, _Options>>
    : traits<Sophus::SE2Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<Eigen::Matrix<Scalar, 2, 1>, _Options> TranslationType;
  typedef Map<Sophus::SO2Group<Scalar>, _Options> SO2Type;
};

template <typename _Scalar, int _Options>
struct traits<Map<const Sophus::SE2Group<_Scalar>, _Options>>
    : traits<const Sophus::SE2Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<const Eigen::Matrix<Scalar, 2, 1>, _Options> TranslationType;
  typedef Map<const Sophus::SO2Group<Scalar>, _Options> SO2Type;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {

// SE2 base type - implements SE2 class but is storage agnostic.
//
// SE(2) is the group of rotations  and translation in 2d. It is the semi-direct
// product of SO(2) and the 2d Euclidean vector space.  The class is represented
// using a composition of SO2Group  for rotation and a 2-vector for translation.
//
// SE(2) is neither compact, nor a commutative group.
//
// See SO2Group for more details of the rotation representation in 2d.
//
template <typename Derived>
class SE2GroupBase {
 public:
  typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;
  typedef typename Eigen::internal::traits<Derived>::TranslationType&
      TranslationReference;
  typedef const typename Eigen::internal::traits<Derived>::TranslationType&
      ConstTranslationReference;
  typedef typename Eigen::internal::traits<Derived>::SO2Type& SO2Reference;
  typedef const typename Eigen::internal::traits<Derived>::SO2Type&
      ConstSO2Reference;

  // Degrees of freedom of manifold, number of dimensions in tangent space
  // (two for translation, three for rotation).
  static const int DoF = 3;
  // Number of internal parameters used (tuple for complex, two for
  // translation).
  static const int num_parameters = 4;
  // Group transformations are 3x3 matrices.
  static const int N = 3;
  typedef Eigen::Matrix<Scalar, N, N> Transformation;
  typedef Eigen::Matrix<Scalar, 2, 1> Point;
  typedef Eigen::Matrix<Scalar, DoF, 1> Tangent;
  typedef Eigen::Matrix<Scalar, DoF, DoF> Adjoint;

  // Adjoint transformation
  //
  // This function return the adjoint transformation ``Ad`` of the group
  // element ``A`` such that for all ``x`` it holds that
  // ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  //
  SOPHUS_FUNC Adjoint Adj() const {
    const Eigen::Matrix<Scalar, 2, 2>& R = so2().matrix();
    Transformation res;
    res.setIdentity();
    res.template topLeftCorner<2, 2>() = R;
    res(0, 2) = translation()[1];
    res(1, 2) = -translation()[0];
    return res;
  }

  // Returns copy of instance casted to NewScalarType.
  //
  template <typename NewScalarType>
  SOPHUS_FUNC SE2Group<NewScalarType> cast() const {
    return SE2Group<NewScalarType>(
        so2().template cast<NewScalarType>(),
        translation().template cast<NewScalarType>());
  }

  // Returns group inverse.
  //
  SOPHUS_FUNC SE2Group<Scalar> inverse() const {
    const SO2Group<Scalar> invR = so2().inverse();
    return SE2Group<Scalar>(invR,
                            invR * (translation() * static_cast<Scalar>(-1)));
  }

  // Logarithmic map
  //
  // Returns tangent space representation (= twist) of the instance.
  //
  SOPHUS_FUNC Tangent log() const { return log(*this); }

  /**
   * \brief Normalize SO2 element
   *
   * It re-normalizes the SO2 element.
   */
  SOPHUS_FUNC void normalize() { so2().normalize(); }

  // Returns 3x3 matrix representation of the instance.
  //
  // It has the following form:
  //
  //   | R t |
  //   | o 1 |
  //
  // where ``R`` is a 2x2 rotation matrix, ``t`` a translation 2-vector and
  // ``o`` a 2-column vector of zeros.
  //
  SOPHUS_FUNC Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.setIdentity();
    homogenious_matrix.block(0, 0, 2, 2) = rotationMatrix();
    homogenious_matrix.col(2).head(2) = translation();
    return homogenious_matrix;
  }

  // Returns the significant first two rows of the matrix above.
  //
  SOPHUS_FUNC Eigen::Matrix<Scalar, 2, 3> matrix2x3() const {
    Eigen::Matrix<Scalar, 2, 3> matrix;
    matrix.block(0, 0, 2, 2) = rotationMatrix();
    matrix.col(2) = translation();
    return matrix;
  }

  // Assignment operator.
  //
  template <typename OtherDerived>
  SOPHUS_FUNC SE2GroupBase<Derived>& operator=(
      const SE2GroupBase<OtherDerived>& other) {
    so2() = other.so2();
    translation() = other.translation();
    return *this;
  }

  // Group multiplication, which is rotation concatenation.
  //
  SOPHUS_FUNC SE2Group<Scalar> operator*(const SE2Group<Scalar>& other) const {
    SE2Group<Scalar> result(*this);
    result *= other;
    return result;
  }

  // Group action on 2-points.
  //
  // This function rotates and translates a two dimensional point ``p`` by the
  // SE(2) element ``bar_T_foo = (bar_R_foo, t_bar)`` (= rigid body
  // transformation):
  //
  //   ``p_bar = bar_R_foo * p_foo + t_bar``.
  //
  SOPHUS_FUNC Point operator*(const Point& p) const {
    return so2() * p + translation();
  }

  // In-place group multiplication.
  //
  SOPHUS_FUNC SE2GroupBase<Derived>& operator*=(const SE2Group<Scalar>& other) {
    translation() += so2() * (other.translation());
    so2() *= other.so2();
    return *this;
  }

  // Returns rotation matrix.
  //
  SOPHUS_FUNC Eigen::Matrix<Scalar, 2, 2> rotationMatrix() const {
    return so2().matrix();
  }

  // Takes in complex number, and normalizes it.
  //
  // Precondition: The complex number must not be close to zero.
  //
  SOPHUS_FUNC void setComplex(const Eigen::Matrix<Scalar, 2, 1>& complex) {
    return so2().setComplex(complex);
  }

  // Sets ``so3`` using ``rotation_matrix``.
  //
  // Precondition: ``R`` must be orthogonal and ``det(R)=1``.
  //
  SOPHUS_FUNC void setRotationMatrix(const Eigen::Matrix<Scalar, 2, 2>& R) {
    so2().setComplex(static_cast<Scalar>(0.5) * (R(0, 0) + R(1, 1)),
                     static_cast<Scalar>(0.5) * (R(1, 0) - R(0, 1)));
  }

  // Mutator of SO3 group.
  //
  SOPHUS_FUNC
  SO2Reference so2() { return static_cast<Derived*>(this)->so2(); }

  // Accessor of SO3 group.
  //
  SOPHUS_FUNC
  ConstSO2Reference so2() const {
    return static_cast<const Derived*>(this)->so2();
  }

  // Mutator of translation vector.
  //
  SOPHUS_FUNC
  TranslationReference translation() {
    return static_cast<Derived*>(this)->translation();
  }

  // Accessor of translation vector
  //
  SOPHUS_FUNC
  ConstTranslationReference translation() const {
    return static_cast<const Derived*>(this)->translation();
  }

  // Accessor of unit complex number.
  //
  SOPHUS_FUNC
  typename Eigen::internal::traits<Derived>::SO2Type::ConstComplexReference
  unit_complex() const {
    return so2().unit_complex();
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  // Derivative of Lie bracket with respect to first element.
  //
  // This function returns ``D_a [a, b]`` with ``D_a`` being the
  // differential operator with respect to ``a``, ``[a, b]`` being the lie
  // bracket of the Lie algebra se3.
  // See ``lieBracket()`` below.
  //
  SOPHUS_FUNC static Transformation d_lieBracketab_by_d_a(const Tangent& b) {
    static const Scalar zero = static_cast<Scalar>(0);
    Eigen::Matrix<Scalar, 2, 1> upsilon2 = b.template head<2>();
    Scalar theta2 = b[2];

    Transformation res;
    res << zero, theta2, -upsilon2[1], -theta2, zero, upsilon2[0], zero, zero,
        zero;
    return res;
  }

  // Group exponential
  //
  // This functions takes in an element of tangent space (= twist ``a``) and
  // returns the corresponding element of the group SE(2).
  //
  // The first two components of ``a`` represent the translational part
  // ``upsilon`` in the tangent space of SE(2), while the last three components
  // of ``a`` represents the rotation vector ``omega``.
  // To be more specific, this function computes ``expmat(hat(a))`` with
  // ``expmat(.)`` being the matrix exponential and ``hat(.)`` the hat-operator
  // of SE(2), see below.
  //
  SOPHUS_FUNC static SE2Group<Scalar> exp(const Tangent& a) {
    Scalar theta = a[2];
    SO2Group<Scalar> so2 = SO2Group<Scalar>::exp(theta);
    Scalar sin_theta_by_theta;
    Scalar one_minus_cos_theta_by_theta;

    if (std::abs(theta) < Constants<Scalar>::epsilon()) {
      Scalar theta_sq = theta * theta;
      sin_theta_by_theta =
          static_cast<Scalar>(1.) - static_cast<Scalar>(1. / 6.) * theta_sq;
      one_minus_cos_theta_by_theta =
          static_cast<Scalar>(0.5) * theta -
          static_cast<Scalar>(1. / 24.) * theta * theta_sq;
    } else {
      sin_theta_by_theta = so2.unit_complex().y() / theta;
      one_minus_cos_theta_by_theta =
          (static_cast<Scalar>(1.) - so2.unit_complex().x()) / theta;
    }
    Eigen::Matrix<Scalar, 2, 1> trans(
        sin_theta_by_theta * a[0] - one_minus_cos_theta_by_theta * a[1],
        one_minus_cos_theta_by_theta * a[0] + sin_theta_by_theta * a[1]);
    return SE2Group<Scalar>(so2, trans);
  }

  // Returns the ith infinitesimal generators of SE(2).
  //
  // The infinitesimal generators of SE(2) are:
  //
  //         |  0  0  1 |
  //   G_0 = |  0  0  0 |
  //         |  0  0  0 |
  //
  //         |  0  0  0 |
  //   G_1 = |  0  0  1 |
  //         |  0  0  0 |
  //
  //         |  0 -1  0 |
  //   G_2 = |  1  0  0 |
  //         |  0  0  0 |
  // Precondition: ``i`` must be in 0, 1 or 2.
  //
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 || i <= 2, "i should be in range [0,2].");
    Tangent e;
    e.setZero();
    e[i] = static_cast<Scalar>(1);
    return hat(e);
  }

  // hat-operator
  //
  // It takes in the 3-vector representation (= twist) and returns the
  // corresponding matrix representation of Lie algebra element.
  //
  // Formally, the ``hat()`` operator of SE(3) is defined as
  //
  //   ``hat(.): R^3 -> R^{3x33},  hat(a) = sum_i a_i * G_i``  (for i=0,1,2)
  //
  // with ``G_i`` being the ith infinitesimal generator of SE(2).
  //
  SOPHUS_FUNC static Transformation hat(const Tangent& a) {
    Transformation Omega;
    Omega.setZero();
    Omega.template topLeftCorner<2, 2>() = SO2Group<Scalar>::hat(a[2]);
    Omega.col(2).template head<2>() = a.template head<2>();
    return Omega;
  }

  // Lie bracket
  //
  // It computes the Lie bracket of SE(2). To be more specific, it computes
  //
  //   ``[omega_1, omega_2]_se2 := vee([hat(omega_1), hat(omega_2)])``
  //
  // with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.) the
  // hat-operator and ``vee(.)`` the vee-operator of SE(2).
  //
  SOPHUS_FUNC static Tangent lieBracket(const Tangent& a, const Tangent& b) {
    Eigen::Matrix<Scalar, 2, 1> upsilon1 = a.template head<2>();
    Eigen::Matrix<Scalar, 2, 1> upsilon2 = b.template head<2>();
    Scalar theta1 = a[2];
    Scalar theta2 = b[2];

    return Tangent(-theta1 * upsilon2[1] + theta2 * upsilon1[1],
                   theta1 * upsilon2[0] - theta2 * upsilon1[0],
                   static_cast<Scalar>(0));
  }

  // Logarithmic map
  //
  // Computes the logarithm, the inverse of the group exponential which maps
  // element of the group (rigid body transformations) to elements of the
  // tangent space (twist).
  //
  // To be specific, this function computes ``vee(logmat(.))`` with
  // ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  // of SE(2).
  //
  SOPHUS_FUNC static Tangent log(const SE2Group<Scalar>& other) {
    Tangent upsilon_theta;
    const SO2Group<Scalar>& so2 = other.so2();
    Scalar theta = SO2Group<Scalar>::log(so2);
    upsilon_theta[2] = theta;
    Scalar halftheta = static_cast<Scalar>(0.5) * theta;
    Scalar halftheta_by_tan_of_halftheta;

    const Eigen::Matrix<Scalar, 2, 1>& z = so2.unit_complex();
    Scalar real_minus_one = z.x() - static_cast<Scalar>(1.);
    if (std::abs(real_minus_one) < Constants<Scalar>::epsilon()) {
      halftheta_by_tan_of_halftheta =
          static_cast<Scalar>(1.) -
          static_cast<Scalar>(1. / 12) * theta * theta;
    } else {
      halftheta_by_tan_of_halftheta = -(halftheta * z.y()) / (real_minus_one);
    }
    Eigen::Matrix<Scalar, 2, 2> V_inv;
    V_inv << halftheta_by_tan_of_halftheta, halftheta, -halftheta,
        halftheta_by_tan_of_halftheta;
    upsilon_theta.template head<2>() = V_inv * other.translation();
    return upsilon_theta;
  }

  // vee-operator
  //
  // It takes the 3x3-matrix representation ``Omega`` and maps it to the
  // corresponding 3-vector representation of Lie algebra.
  //
  // This is the inverse of the hat-operator, see above.
  //
  SOPHUS_FUNC static Tangent vee(const Transformation& Omega) {
    Tangent upsilon_omega;
    upsilon_omega.template head<2>() = Omega.col(2).template head<2>();
    upsilon_omega[2] =
        SO2Group<Scalar>::vee(Omega.template topLeftCorner<2, 2>());
    return upsilon_omega;
  }
};

// SE2 default type - Constructors and default storage for SE3 Type.
template <typename _Scalar, int _Options>
class SE2Group : public SE2GroupBase<SE2Group<_Scalar, _Options>> {
  typedef SE2GroupBase<SE2Group<_Scalar, _Options>> Base;

 public:
  typedef typename Eigen::internal::traits<SE2Group<_Scalar, _Options>>::Scalar
      Scalar;
  typedef typename Eigen::internal::traits<
      SE2Group<_Scalar, _Options>>::TranslationType& TranslationReference;
  typedef const typename Eigen::internal::traits<
      SE2Group<_Scalar, _Options>>::TranslationType& ConstTranslationReference;
  typedef
      typename Eigen::internal::traits<SE2Group<_Scalar, _Options>>::SO2Type&
          SO2Reference;
  typedef const typename Eigen::internal::traits<
      SE2Group<_Scalar, _Options>>::SO2Type& ConstSO2Reference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor initialize rigid body motion to the identity.
  //
  SOPHUS_FUNC SE2Group() : translation_(Eigen::Matrix<Scalar, 2, 1>::Zero()) {}

  // Copy constructor
  //
  template <typename OtherDerived>
  SOPHUS_FUNC SE2Group(const SE2GroupBase<OtherDerived>& other)
      : so2_(other.so2()), translation_(other.translation()) {}

  // Constructor from SO3 and translation vector
  //
  template <typename OtherDerived>
  SOPHUS_FUNC SE2Group(const SO2GroupBase<OtherDerived>& so2,
                       const Point& translation)
      : so2_(so2), translation_(translation) {}

  // Constructor from rotation matrix and translation vector
  //
  // Precondition: Rotation matrix needs to be orthogonal with determinant of 1.
  //
  SOPHUS_FUNC
  SE2Group(const typename SO2Group<Scalar>::Transformation& rotation_matrix,
           const Point& translation)
      : so2_(rotation_matrix), translation_(translation) {}

  // Constructor from rotation angle and translation vector.
  //
  SOPHUS_FUNC SE2Group(const Scalar& theta, const Point& translation)
      : so2_(theta), translation_(translation) {}

  // Constructor from complex number and translation vector
  //
  // Precondition: ``complex` must not be close to zero.
  SOPHUS_FUNC SE2Group(const std::complex<Scalar>& complex,
                       const Point& translation)
      : so2_(complex), translation_(translation) {}

  // Constructor from 3x3 matrix
  //
  // Precondition: Rotation matrix needs to be orthogonal with determinant of 1.
  //               The last row must be (0, 0, 1).
  //
  SOPHUS_FUNC explicit SE2Group(const Transformation& T)
      : so2_(T.template topLeftCorner<2, 2>().eval()),
        translation_(T.template block<2, 1>(0, 2)) {}

  // This provides unsafe read/write access to internal data. SO(2) is
  // represented by a complex number (two parameters). When using direct write
  // access, the user needs to take care of that the complex number stays
  // normalized.
  //
  SOPHUS_FUNC Scalar* data() {
    // so2_ and translation_ are layed out sequentially with no padding
    return so2_.data();
  }

  // Const version of data() above.
  //
  SOPHUS_FUNC const Scalar* data() const {
    // so2_ and translation_ are layed out sequentially with no padding
    return so2_.data();
  }

  // Accessor of SO3
  //
  SOPHUS_FUNC SO2Reference so2() { return so2_; }

  // Mutator of SO3
  //
  SOPHUS_FUNC ConstSO2Reference so2() const { return so2_; }

  // Mutator of translation vector
  //
  SOPHUS_FUNC TranslationReference translation() { return translation_; }

  // Accessor of translation vector
  //
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return translation_;
  }

 protected:
  Sophus::SO2Group<Scalar> so2_;
  Eigen::Matrix<Scalar, 2, 1> translation_;
};

}  // end namespace

namespace Eigen {

// Specialization of Eigen::Map for ``SE2GroupBase``.
//
// Allows us to wrap SE2 objects around POD array.
template <typename _Scalar, int _Options>
class Map<Sophus::SE2Group<_Scalar>, _Options>
    : public Sophus::SE2GroupBase<Map<Sophus::SE2Group<_Scalar>, _Options>> {
  typedef Sophus::SE2GroupBase<Map<Sophus::SE2Group<_Scalar>, _Options>> Base;

 public:
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  typedef typename Eigen::internal::traits<Map>::TranslationType&
      TranslationReference;
  typedef const typename Eigen::internal::traits<Map>::TranslationType&
      ConstTranslationReference;
  typedef typename Eigen::internal::traits<Map>::SO2Type& SO2Reference;

  typedef const typename Eigen::internal::traits<Map>::SO2Type&
      ConstSO2Reference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  Map(Scalar* coeffs)
      : so2_(coeffs),
        translation_(coeffs + Sophus::SO2Group<Scalar>::num_parameters) {}

  // Mutator of SO3
  //
  SOPHUS_FUNC SO2Reference so2() { return so2_; }

  // Accessor of SO3
  //
  SOPHUS_FUNC ConstSO2Reference so2() const { return so2_; }

  // Mutator of translation vector
  //
  SOPHUS_FUNC TranslationReference translation() { return translation_; }

  // Accessor of translation vector
  //
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return translation_;
  }

 protected:
  Map<Sophus::SO2Group<Scalar>, _Options> so2_;
  Map<Eigen::Matrix<Scalar, 2, 1>, _Options> translation_;
};

// Specialization of Eigen::Map for ``const SE2GroupBase``/
//
// Allows us to wrap SE2 objects around POD array.
template <typename _Scalar, int _Options>
class Map<const Sophus::SE2Group<_Scalar>, _Options>
    : public Sophus::SE2GroupBase<
          Map<const Sophus::SE2Group<_Scalar>, _Options>> {
  typedef Sophus::SE2GroupBase<Map<const Sophus::SE2Group<_Scalar>, _Options>>
      Base;

 public:
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  typedef const typename Eigen::internal::traits<Map>::TranslationType&
      ConstTranslationReference;
  typedef const typename Eigen::internal::traits<Map>::SO2Type&
      ConstSO2Reference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(const Scalar* coeffs)
      : so2_(coeffs),
        translation_(coeffs + Sophus::SO2Group<Scalar>::num_parameters) {}

  SOPHUS_FUNC Map(const Scalar* trans_coeffs, const Scalar* rot_coeffs)
      : so2_(rot_coeffs), translation_(trans_coeffs) {}

  // Accessor of SO3
  //
  SOPHUS_FUNC ConstSO2Reference so2() const { return so2_; }

  // Accessor of translation vector
  //
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return translation_;
  }

 protected:
  const Map<const Sophus::SO2Group<Scalar>, _Options> so2_;
  const Map<const Eigen::Matrix<Scalar, 2, 1>, _Options> translation_;
};
}

#endif
