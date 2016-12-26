// This file is part of Sophus.
//
// Copyright 2011-2013 Hauke Strasdat
//           2012-2013 Steven Lovegrove
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

#ifndef SOPHUS_SE3_HPP
#define SOPHUS_SE3_HPP

#include "so3.hpp"

namespace Sophus {
template <typename _Scalar, int _Options = 0>
class SE3Group;
typedef SE3Group<double> SE3d; /**< double precision SE3 */
typedef SE3Group<float> SE3f;  /**< single precision SE3 */
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;
}  // namespace Sophus

namespace Eigen {
namespace internal {

template <typename _Scalar, int _Options>
struct traits<Sophus::SE3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Eigen::Matrix<Scalar, 3, 1> TranslationType;
  typedef Sophus::SO3Group<Scalar> SO3Type;
};

template <typename _Scalar, int _Options>
struct traits<Map<Sophus::SE3Group<_Scalar>, _Options>>
    : traits<Sophus::SE3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<Eigen::Matrix<Scalar, 3, 1>, _Options> TranslationType;
  typedef Map<Sophus::SO3Group<Scalar>, _Options> SO3Type;
};

template <typename _Scalar, int _Options>
struct traits<Map<const Sophus::SE3Group<_Scalar>, _Options>>
    : traits<const Sophus::SE3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<const Eigen::Matrix<Scalar, 3, 1>, _Options> TranslationType;
  typedef Map<const Sophus::SO3Group<Scalar>, _Options> SO3Type;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {

// SE3 base type - implements SE3 class but is storage agnostic.
//
// SE(3) is the group of rotations  and translation in 3d. It is the semi-direct
// product of SO(3) and the 3d Euclidean vector space.  The class is represented
// using a composition of SO3Group  for rotation and a one 3-vector for
// translation.
//
// SE(3) is neither compact, nor a commutative group.
//
// See SO3Group for more details of the rotation representation in 3d.
//
template <typename Derived>
class SE3GroupBase {
 public:
  typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;
  typedef typename Eigen::internal::traits<Derived>::TranslationType&
      TranslationReference;
  typedef const typename Eigen::internal::traits<Derived>::TranslationType&
      ConstTranslationReference;
  typedef typename Eigen::internal::traits<Derived>::SO3Type& SO3Reference;
  typedef const typename Eigen::internal::traits<Derived>::SO3Type&
      ConstSO3Reference;

  // Degrees of freedom of manifold, number of dimensions in tangent space
  // (two for translation, two for rotation).
  static const int DoF = 6;
  // Number of internal parameters used (4-tuple for quaternion, three for
  // translation).
  static const int num_parameters = 7;
  // Group transformations are 4x4 matrices.
  static const int N = 4;
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
  SOPHUS_FUNC Adjoint Adj() const {
    const Eigen::Matrix<Scalar, 3, 3>& R = so3().matrix();
    Adjoint res;
    res.block(0, 0, 3, 3) = R;
    res.block(3, 3, 3, 3) = R;
    res.block(0, 3, 3, 3) = SO3Group<Scalar>::hat(translation()) * R;
    res.block(3, 0, 3, 3) = Eigen::Matrix<Scalar, 3, 3>::Zero(3, 3);
    return res;
  }

  // Returns Affine3 representation.
  //
  SOPHUS_FUNC
  Eigen::Transform<Scalar, 3, Eigen::Affine> affine3() const {
    return Eigen::Transform<Scalar, 3, Eigen::Affine>(matrix());
  }

  // Returns copy of instance casted to NewScalarType.
  //
  template <typename NewScalarType>
  SOPHUS_FUNC SE3Group<NewScalarType> cast() const {
    return SE3Group<NewScalarType>(
        so3().template cast<NewScalarType>(),
        translation().template cast<NewScalarType>());
  }

  // Returns ``*this`` times the ith generator of internal representation.
  //
  SOPHUS_FUNC Eigen::Matrix<Scalar, num_parameters, 1>
  internalMultiplyByGenerator(int i) const {
    Eigen::Matrix<Scalar, num_parameters, 1> res;

    Eigen::Quaternion<Scalar> internal_gen_q;
    Eigen::Matrix<Scalar, 3, 1> internal_gen_t;

    internalGenerator(i, &internal_gen_q, &internal_gen_t);

    res.template head<4>() = (unit_quaternion() * internal_gen_q).coeffs();
    res.template tail<3>() = unit_quaternion() * internal_gen_t;
    return res;
  }

  // Returns Jacobian of generator of internal SU(2) representation.
  //
  SOPHUS_FUNC Eigen::Matrix<Scalar, num_parameters, DoF> internalJacobian()
      const {
    Eigen::Matrix<Scalar, num_parameters, DoF> J;
    for (int i = 0; i < DoF; ++i) {
      J.col(i) = internalMultiplyByGenerator(i);
    }
    return J;
  }

  // Returns group inverse.
  //
  SOPHUS_FUNC SE3Group<Scalar> inverse() const {
    SO3Group<Scalar> invR = so3().inverse();
    return SE3Group<Scalar>(invR,
                            invR * (translation() * static_cast<Scalar>(-1)));
  }

  // Logarithmic map
  //
  // Returns tangent space representation (= twist) of the instance.
  //
  SOPHUS_FUNC Tangent log() const { return log(*this); }

  // It re-normalizes the SO3 element.
  //
  // Note: Because of the class invariant of SO3, there is typically no need to
  // call this function directly.
  //
  SOPHUS_FUNC void normalize() { so3().normalize(); }

  // Returns 4x4 matrix representation of the instance.
  //
  // It has the following form:
  //
  //   | R t |
  //   | o 1 |
  //
  // where ``R`` is a 3x3 rotation matrix, ``t`` a translation 3-vector and
  // ``o`` a 3-column vector of zeros.
  //
  SOPHUS_FUNC Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.setIdentity();
    homogenious_matrix.block(0, 0, 3, 3) = rotationMatrix();
    homogenious_matrix.col(3).head(3) = translation();
    return homogenious_matrix;
  }

  // Returns the significant first three rows of the matrix above.
  //
  SOPHUS_FUNC Eigen::Matrix<Scalar, 3, 4> matrix3x4() const {
    Eigen::Matrix<Scalar, 3, 4> matrix;
    matrix.block(0, 0, 3, 3) = rotationMatrix();
    matrix.col(3) = translation();
    return matrix;
  }

  // Assignment operator.
  //
  template <typename OtherDerived>
  SOPHUS_FUNC SE3GroupBase<Derived>& operator=(
      const SE3GroupBase<OtherDerived>& other) {
    so3() = other.so3();
    translation() = other.translation();
    return *this;
  }

  // Group multiplication, which is rotation concatenation.
  //
  SOPHUS_FUNC SE3Group<Scalar> operator*(const SE3Group<Scalar>& other) const {
    SE3Group<Scalar> result(*this);
    result *= other;
    return result;
  }

  // Group action on 3-points.
  //
  // This function rotates and translates a three dimensional point ``p`` by the
  // SE(3) element ``bar_T_foo = (bar_R_foo, t_bar)`` (= rigid body
  // transformation):
  //
  //   ``p_bar = bar_R_foo * p_foo + t_bar``.
  //
  SOPHUS_FUNC Point operator*(const Point& p) const {
    return so3() * p + translation();
  }

  // In-place group multiplication.
  //
  SOPHUS_FUNC SE3GroupBase<Derived>& operator*=(const SE3Group<Scalar>& other) {
    translation() += so3() * (other.translation());
    so3() *= other.so3();
    return *this;
  }

  // Returns rotation matrix.
  //
  SOPHUS_FUNC Eigen::Matrix<Scalar, 3, 3> rotationMatrix() const {
    return so3().matrix();
  }

  // Mutator of SO3 group.
  //
  SOPHUS_FUNC SO3Reference so3() { return static_cast<Derived*>(this)->so3(); }

  // Accessor of SO3 group.
  //
  SOPHUS_FUNC ConstSO3Reference so3() const {
    return static_cast<const Derived*>(this)->so3();
  }

  // Setter using Affine3
  //
  // Precondition: 3x3 sub-matrix needs to be orthogonal with determinant of 1.
  //
  SOPHUS_FUNC void setAffine3(
      const Eigen::Transform<Scalar, 3, Eigen::Affine>& affine3) {
    so3().setRotationMatrix(affine3.matrix().template topLeftCorner<3, 3>());
    translation() = affine3.matrix().template topRightCorner<3, 1>();
  }

  // Takes in quaternion, and normalizes it.
  //
  // Precondition: The quaternion must not be close to zero.
  //
  SOPHUS_FUNC void setQuaternion(const Eigen::Quaternion<Scalar>& quat) {
    so3().setQuaternion(quat);
  }

  // Sets ``so3`` using ``rotation_matrix``.
  //
  // Precondition: ``R`` must be orthogonal and ``det(R)=1``.
  //
  SOPHUS_FUNC void setRotationMatrix(
      const Eigen::Matrix<Scalar, 3, 3>& rotation_matrix) {
    so3().setQuaternion(Eigen::Quaternion<Scalar>(rotation_matrix));
  }

  // Mutator of translation vector.
  //
  SOPHUS_FUNC TranslationReference translation() {
    return static_cast<Derived*>(this)->translation();
  }

  // Accessor of translation vector
  //
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return static_cast<const Derived*>(this)->translation();
  }

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC typename Eigen::internal::traits<
      Derived>::SO3Type::ConstQuaternionReference
  unit_quaternion() const {
    return so3().unit_quaternion();
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  // Derivative of Lie bracket with respect to first element.
  //
  // This function returns ``D_a [a, b]`` with ``D_a`` being the
  // differential operator with respect to ``a``, ``[a, b]`` being the lie
  // bracket of the Lie algebra se(3).
  // See ``lieBracket()`` below.
  //
  SOPHUS_FUNC static Adjoint d_lieBracketab_by_d_a(const Tangent& b) {
    Adjoint res;
    res.setZero();

    const Eigen::Matrix<Scalar, 3, 1>& upsilon2 = b.template head<3>();
    const Eigen::Matrix<Scalar, 3, 1>& omega2 = b.template tail<3>();

    res.template topLeftCorner<3, 3>() = -SO3Group<Scalar>::hat(omega2);
    res.template topRightCorner<3, 3>() = -SO3Group<Scalar>::hat(upsilon2);
    res.template bottomRightCorner<3, 3>() = -SO3Group<Scalar>::hat(omega2);
    return res;
  }

  // Group exponential
  //
  // This functions takes in an element of tangent space (= twist ``a``) and
  // returns the corresponding element of the group SE(3).
  //
  // The first three components of ``a`` represent the translational part
  // ``upsilon`` in the tangent space of SE(3), while the last three components
  // of ``a`` represents the rotation vector ``omega``.
  // To be more specific, this function computes ``expmat(hat(a))`` with
  // ``expmat(.)`` being the matrix exponential and ``hat(.)`` the hat-operator
  // of SE(3), see below.
  //
  SOPHUS_FUNC static SE3Group<Scalar> exp(const Tangent& a) {
    using std::cos;
    using std::sin;
    const Eigen::Matrix<Scalar, 3, 1> omega = a.template tail<3>();

    Scalar theta;
    SO3Group<Scalar> so3 = SO3Group<Scalar>::expAndTheta(omega, &theta);
    Eigen::Matrix<Scalar, 3, 3> Omega = SO3Group<Scalar>::hat(omega);
    Eigen::Matrix<Scalar, 3, 3> Omega_sq = Omega * Omega;
    Eigen::Matrix<Scalar, 3, 3> V;

    if (theta < Constants<Scalar>::epsilon()) {
      V = so3.matrix();
      // Note: That is an accurate expansion!
    } else {
      Scalar theta_sq = theta * theta;
      V = (Eigen::Matrix<Scalar, 3, 3>::Identity() +
           (static_cast<Scalar>(1) - cos(theta)) / (theta_sq)*Omega +
           (theta - sin(theta)) / (theta_sq * theta) * Omega_sq);
    }
    return SE3Group<Scalar>(so3, V * a.template head<3>());
  }

  // Returns the ith infinitesimal generators of SE(3).
  //
  // The infinitesimal generators of SE(3) are:
  //
  //         |  0  0  0  1 |
  //   G_0 = |  0  0  0  0 |
  //         |  0  0  0  0 |
  //         |  0  0  0  0 |
  //
  //         |  0  0  0  0 |
  //   G_1 = |  0  0  0  1 |
  //         |  0  0  0  0 |
  //         |  0  0  0  0 |
  //
  //         |  0  0  0  0 |
  //   G_2 = |  0  0  0  0 |
  //         |  0  0  0  1 |
  //         |  0  0  0  0 |
  //
  //         |  0  0  0  0 |
  //   G_3 = |  0  0 -1  0 |
  //         |  0  1  0  0 |
  //         |  0  0  0  0 |
  //
  //         |  0  0  1  0 |
  //   G_4 = |  0  0  0  0 |
  //         | -1  0  0  0 |
  //         |  0  0  0  0 |
  //
  //         |  0 -1  0  0 |
  //   G_5 = |  1  0  0  0 |
  //         |  0  0  0  0 |
  //         |  0  0  0  0 |
  //
  // Precondition: ``i`` must be in [0, 5].
  //
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 && i <= 5, "i should be in range [0,5].");
    Tangent e;
    e.setZero();
    e[i] = static_cast<Scalar>(1);
    return hat(e);
  }

  // Returns the ith generator of internal representation.
  //
  // Precondition: ``i`` must be in [0, 5].
  //
  SOPHUS_FUNC static void internalGenerator(
      int i, Eigen::Quaternion<Scalar>* internal_gen_q,
      Eigen::Matrix<Scalar, 3, 1>* internal_gen_t) {
    SOPHUS_ENSURE(i >= 0 && i <= 5, "i should be in range [0,5]");
    SOPHUS_ENSURE(internal_gen_q != NULL,
                  "internal_gen_q must not be the null pointer");
    SOPHUS_ENSURE(internal_gen_t != NULL,
                  "internal_gen_t must not be the null pointer");

    internal_gen_q->coeffs().setZero();
    internal_gen_t->setZero();
    if (i < 3) {
      (*internal_gen_t)[i] = static_cast<Scalar>(1);
    } else {
      SO3Group<Scalar>::internalGenerator(i - 3, internal_gen_q);
      ;
    }
  }

  // hat-operator
  //
  // It takes in the 6-vector representation (= twist) and returns the
  // corresponding matrix representation of Lie algebra element.
  //
  // Formally, the ``hat()`` operator of SE(3) is defined as
  //
  //   ``hat(.): R^6 -> R^{4x4},  hat(a) = sum_i a_i * G_i``  (for i=0,...,5)
  //
  // with ``G_i`` being the ith infinitesimal generator of SE(3).
  //
  SOPHUS_FUNC static Transformation hat(const Tangent& a) {
    Transformation Omega;
    Omega.setZero();
    Omega.template topLeftCorner<3, 3>() =
        SO3Group<Scalar>::hat(a.template tail<3>());
    Omega.col(3).template head<3>() = a.template head<3>();
    return Omega;
  }

  // Lie bracket
  //
  // It computes the Lie bracket of SE(3). To be more specific, it computes
  //
  //   ``[omega_1, omega_2]_se3 := vee([hat(omega_1), hat(omega_2)])``
  //
  // with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.) the
  // hat-operator and ``vee(.)`` the vee-operator of SE(3).
  //
  SOPHUS_FUNC static Tangent lieBracket(const Tangent& a, const Tangent& b) {
    const Eigen::Matrix<Scalar, 3, 1>& upsilon1 = a.template head<3>();
    const Eigen::Matrix<Scalar, 3, 1>& upsilon2 = b.template head<3>();
    Eigen::Matrix<Scalar, 3, 1> omega1 = a.template tail<3>();
    Eigen::Matrix<Scalar, 3, 1> omega2 = b.template tail<3>();

    Tangent res;
    res.template head<3>() = omega1.cross(upsilon2) + upsilon1.cross(omega2);
    res.template tail<3>() = omega1.cross(omega2);

    return res;
  }

  // Logarithmic map
  //
  // Computes the logarithm, the inverse of the group exponential which maps
  // element of the group (rigid body transformations) to elements of the
  // tangent space (twist).
  //
  // To be specific, this function computes ``vee(logmat(.))`` with
  // ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  // of SE(3).
  //
  SOPHUS_FUNC static Tangent log(const SE3Group<Scalar>& se3) {
    using std::abs;
    Tangent upsilon_omega;
    Scalar theta;
    upsilon_omega.template tail<3>() =
        SO3Group<Scalar>::logAndTheta(se3.so3(), &theta);

    if (abs(theta) < Constants<Scalar>::epsilon()) {
      Eigen::Matrix<Scalar, 3, 3> Omega =
          SO3Group<Scalar>::hat(upsilon_omega.template tail<3>());
      Eigen::Matrix<Scalar, 3, 3> V_inv =
          Eigen::Matrix<Scalar, 3, 3>::Identity() -
          static_cast<Scalar>(0.5) * Omega +
          static_cast<Scalar>(1. / 12.) * (Omega * Omega);

      upsilon_omega.template head<3>() = V_inv * se3.translation();
    } else {
      Eigen::Matrix<Scalar, 3, 3> Omega =
          SO3Group<Scalar>::hat(upsilon_omega.template tail<3>());
      Eigen::Matrix<Scalar, 3, 3> V_inv =
          (Eigen::Matrix<Scalar, 3, 3>::Identity() -
           static_cast<Scalar>(0.5) * Omega +
           (static_cast<Scalar>(1) -
            theta / (static_cast<Scalar>(2) * tan(theta / Scalar(2)))) /
               (theta * theta) * (Omega * Omega));
      upsilon_omega.template head<3>() = V_inv * se3.translation();
    }
    return upsilon_omega;
  }

  // vee-operator
  //
  // It takes 4x4-matrix representation ``Omega`` and maps it to the
  // corresponding 6-vector representation of Lie algebra.
  //
  // This is the inverse of the hat-operator, see above.
  //
  SOPHUS_FUNC static Tangent vee(const Transformation& Omega) {
    Tangent upsilon_omega;
    upsilon_omega.template head<3>() = Omega.col(3).template head<3>();
    upsilon_omega.template tail<3>() =
        SO3Group<Scalar>::vee(Omega.template topLeftCorner<3, 3>());
    return upsilon_omega;
  }
};

// SE3 default type - Constructors and default storage for SE3 Type.
template <typename _Scalar, int _Options>
class SE3Group : public SE3GroupBase<SE3Group<_Scalar, _Options>> {
  typedef SE3GroupBase<SE3Group<_Scalar, _Options>> Base;

 public:
  typedef typename Eigen::internal::traits<SE3Group<_Scalar, _Options>>::Scalar
      Scalar;
  typedef
      typename Eigen::internal::traits<SE3Group<_Scalar, _Options>>::SO3Type&
          SO3Reference;
  typedef const typename Eigen::internal::traits<
      SE3Group<_Scalar, _Options>>::SO3Type& ConstSO3Reference;
  typedef typename Eigen::internal::traits<
      SE3Group<_Scalar, _Options>>::TranslationType& TranslationReference;
  typedef const typename Eigen::internal::traits<
      SE3Group<_Scalar, _Options>>::TranslationType& ConstTranslationReference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor initialize rigid body motion to the identity.
  //
  SOPHUS_FUNC SE3Group() : translation_(Eigen::Matrix<Scalar, 3, 1>::Zero()) {}

  // Copy constructor
  //
  template <typename OtherDerived>
  SOPHUS_FUNC SE3Group(const SE3GroupBase<OtherDerived>& other)
      : so3_(other.so3()), translation_(other.translation()) {}

  // Constructor from SO3 and translation vector
  //
  template <typename OtherDerived>
  SOPHUS_FUNC SE3Group(const SO3GroupBase<OtherDerived>& so3,
                       const Point& translation)
      : so3_(so3), translation_(translation) {}

  // Constructor from rotation matrix and translation vector
  //
  // Precondition: Rotation matrix needs to be orthogonal with determinant of 1.
  //
  SOPHUS_FUNC
  SE3Group(const Eigen::Matrix<Scalar, 3, 3>& rotation_matrix,
           const Point& translation)
      : so3_(rotation_matrix), translation_(translation) {}

  // Constructor from quaternion and translation vector.
  //
  // Precondition: quaternion must not be close to zero.
  //
  SOPHUS_FUNC SE3Group(const Eigen::Quaternion<Scalar>& quaternion,
                       const Point& translation)
      : so3_(quaternion), translation_(translation) {}

  // Constructor from 4x4 matrix
  //
  // Precondition: Rotation matrix needs to be orthogonal with determinant of 1.
  //               The last row must be (0, 0, 0, 1).
  //
  SOPHUS_FUNC explicit SE3Group(const Eigen::Matrix<Scalar, 4, 4>& T)
      : so3_(T.template topLeftCorner<3, 3>()),
        translation_(T.template block<3, 1>(0, 3)) {
    SOPHUS_ENSURE(
        (T.row(3) - Eigen::Matrix<Scalar, 1, 4>(0, 0, 0, 1)).squaredNorm() <
            Constants<Scalar>::epsilon(),
        "Last row is not (0,0,0,1), but (%).", T.row(3));
  }

  // Constructor from Affine3
  //
  // Precondition: Rotation matrix needs to be orthogonal with determinant of 1.
  //
  SOPHUS_FUNC explicit SE3Group(
      const Eigen::Transform<Scalar, 3, Eigen::Affine>& affine3)
      : so3_(affine3.matrix().template topLeftCorner<3, 3>()),
        translation_(affine3.matrix().template block<3, 1>(0, 3)) {}

  // This provides unsafe read/write access to internal data. SO(3) is
  // represented by an Eigen::Quaternion (four parameters). When using direct
  // write access, the user needs to take care of that the quaternion stays
  // normalized.
  //
  SOPHUS_FUNC Scalar* data() {
    // so3_ and translation_ are laid out sequentially with no padding
    return so3_.data();
  }

  // Const version of data() above.
  //
  SOPHUS_FUNC const Scalar* data() const {
    // so3_ and translation_ are laid out sequentially with no padding
    return so3_.data();
  }

  // Accessor of SO3
  //
  SOPHUS_FUNC SO3Reference so3() { return so3_; }

  // Mutator of SO3
  //
  SOPHUS_FUNC ConstSO3Reference so3() const { return so3_; }

  // Mutator of translation vector
  //
  SOPHUS_FUNC TranslationReference translation() { return translation_; }

  // Accessor of translation vector
  //
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return translation_;
  }

 protected:
  Sophus::SO3Group<Scalar> so3_;
  Eigen::Matrix<Scalar, 3, 1> translation_;
};

}  // namespace Sophus

namespace Eigen {

// Specialization of Eigen::Map for ``SE3GroupBase``.
//
// Allows us to wrap SE3 objects around POD array.
template <typename _Scalar, int _Options>
class Map<Sophus::SE3Group<_Scalar>, _Options>
    : public Sophus::SE3GroupBase<Map<Sophus::SE3Group<_Scalar>, _Options>> {
  typedef Sophus::SE3GroupBase<Map<Sophus::SE3Group<_Scalar>, _Options>> Base;

 public:
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  typedef typename Eigen::internal::traits<Map>::TranslationType&
      TranslationReference;
  typedef const typename Eigen::internal::traits<Map>::TranslationType&
      ConstTranslationReference;
  typedef typename Eigen::internal::traits<Map>::SO3Type& SO3Reference;
  typedef const typename Eigen::internal::traits<Map>::SO3Type&
      ConstSO3Reference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar* coeffs)
      : so3_(coeffs),
        translation_(coeffs + Sophus::SO3Group<Scalar>::num_parameters) {}

  // Mutator of SO3
  //
  SOPHUS_FUNC SO3Reference so3() { return so3_; }

  // Accessor of SO3
  //
  SOPHUS_FUNC ConstSO3Reference so3() const { return so3_; }

  // Mutator of translation vector
  //
  SOPHUS_FUNC TranslationReference translation() { return translation_; }

  // Accessor of translation vector
  //
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return translation_;
  }

 protected:
  Map<Sophus::SO3Group<Scalar>, _Options> so3_;
  Map<Eigen::Matrix<Scalar, 3, 1>, _Options> translation_;
};

// Specialization of Eigen::Map for ``const SE3GroupBase``
//
// Allows us to wrap SE3 objects around POD array.
template <typename _Scalar, int _Options>
class Map<const Sophus::SE3Group<_Scalar>, _Options>
    : public Sophus::SE3GroupBase<
          Map<const Sophus::SE3Group<_Scalar>, _Options>> {
  typedef Sophus::SE3GroupBase<Map<const Sophus::SE3Group<_Scalar>, _Options>>
      Base;

 public:
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  typedef const typename Eigen::internal::traits<Map>::TranslationType&
      ConstTranslationReference;
  typedef const typename Eigen::internal::traits<Map>::SO3Type&
      ConstSO3Reference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(const Scalar* coeffs)
      : so3_(coeffs),
        translation_(coeffs + Sophus::SO3Group<Scalar>::num_parameters) {}

  SOPHUS_FUNC Map(const Scalar* trans_coeffs, const Scalar* rot_coeffs)
      : so3_(rot_coeffs), translation_(trans_coeffs) {}

  // Accessor of SO3
  //
  SOPHUS_FUNC ConstSO3Reference so3() const { return so3_; }

  // Accessor of translation vector
  //
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return translation_;
  }

 protected:
  const Map<const Sophus::SO3Group<Scalar>, _Options> so3_;
  const Map<const Eigen::Matrix<Scalar, 3, 1>, _Options> translation_;
};
}

#endif
