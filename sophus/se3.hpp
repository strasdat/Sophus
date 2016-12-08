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

////////////////////////////////////////////////////////////////////////////
// Forward Declarations / typedefs
////////////////////////////////////////////////////////////////////////////

namespace Sophus {
template <typename _Scalar, int _Options = 0>
class SE3Group;
typedef SE3Group<double> SE3d; /**< double precision SE3 */
typedef SE3Group<float> SE3f;  /**< single precision SE3 */
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;
}

////////////////////////////////////////////////////////////////////////////
// Eigen Traits (For querying derived types in CRTP hierarchy)
////////////////////////////////////////////////////////////////////////////

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
}
}

namespace Sophus {
using std::abs;
using std::cos;
using std::sin;

/**
 * \brief SE3 base type - implements SE3 class but is storage agnostic
 *
 * [add more detailed description/tutorial]
 */
template <typename Derived>
class SE3GroupBase {
 public:
  /** \brief scalar type */
  typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;
  /** \brief translation reference type */
  typedef typename Eigen::internal::traits<Derived>::TranslationType&
      TranslationReference;
  /** \brief translation const reference type */
  typedef const typename Eigen::internal::traits<Derived>::TranslationType&
      ConstTranslationReference;
  /** \brief SO3 reference type */
  typedef typename Eigen::internal::traits<Derived>::SO3Type& SO3Reference;
  /** \brief SO3 const reference type */
  typedef const typename Eigen::internal::traits<Derived>::SO3Type&
      ConstSO3Reference;

  /** \brief degree of freedom of group
    *        (three for translation, three for rotation) */
  static const int DoF = 6;
  /** \brief number of internal parameters used
   *         (unit quaternion for rotation + translation 3-vector) */
  static const int num_parameters = 7;
  /** \brief group transformations are NxN matrices */
  static const int N = 4;
  /** \brief group transfomation type */
  typedef Eigen::Matrix<Scalar, N, N> Transformation;
  /** \brief point type */
  typedef Eigen::Matrix<Scalar, 3, 1> Point;
  /** \brief tangent vector type */
  typedef Eigen::Matrix<Scalar, DoF, 1> Tangent;
  /** \brief adjoint transformation type */
  typedef Eigen::Matrix<Scalar, DoF, DoF> Adjoint;

  /**
   * \brief Adjoint transformation
   *
   * This function return the adjoint transformation \f$ Ad \f$ of the
   * group instance \f$ A \f$  such that for all \f$ x \f$
   * it holds that \f$ \widehat{Ad_A\cdot x} = A\widehat{x}A^{-1} \f$
   * with \f$\ \widehat{\cdot} \f$ being the hat()-operator.
   */
  SOPHUS_FUNC Adjoint Adj() const {
    const Eigen::Matrix<Scalar, 3, 3>& R = so3().matrix();
    Adjoint res;
    res.block(0, 0, 3, 3) = R;
    res.block(3, 3, 3, 3) = R;
    res.block(0, 3, 3, 3) = SO3Group<Scalar>::hat(translation()) * R;
    res.block(3, 0, 3, 3) = Eigen::Matrix<Scalar, 3, 3>::Zero(3, 3);
    return res;
  }

  /**
   * \returns Affine3 transformation
   */
  SOPHUS_FUNC
  Eigen::Transform<Scalar, 3, Eigen::Affine> affine3() const {
    return Eigen::Transform<Scalar, 3, Eigen::Affine>(matrix());
  }

  /**
   * \returns copy of instance casted to NewScalarType
   */
  template <typename NewScalarType>
  SOPHUS_FUNC SE3Group<NewScalarType> cast() const {
    return SE3Group<NewScalarType>(
        so3().template cast<NewScalarType>(),
        translation().template cast<NewScalarType>());
  }

  /**
    * \brief multiply by ith internal generator
    *
    * \returns *this  x  ith generator of internal data representation
    *
    * \see internalGenerator
    */
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

  /**
   * \returns Jacobian of generator of internal data represenation
   *
   * \see internalMultiplyByGenerator
   */
  SOPHUS_FUNC
  Eigen::Matrix<Scalar, num_parameters, DoF> internalJacobian() const {
    Eigen::Matrix<Scalar, num_parameters, DoF> J;
    for (int i = 0; i < DoF; ++i) {
      J.col(i) = internalMultiplyByGenerator(i);
    }
    return J;
  }

  /**
   * \returns Group inverse of instance
   */
  SOPHUS_FUNC SE3Group<Scalar> inverse() const {
    SO3Group<Scalar> invR = so3().inverse();
    return SE3Group<Scalar>(invR,
                            invR * (translation() * static_cast<Scalar>(-1)));
  }

  /**
   * \brief Logarithmic map
   *
   * \returns tangent space representation
   *          (translational part and rotation vector) of instance
   *
   * \see  log().
   */
  SOPHUS_FUNC Tangent log() const { return log(*this); }

  /**
   * \brief Normalize SO3 element
   *
   * It re-normalizes the SO3 element.
   */
  SOPHUS_FUNC void normalize() { so3().normalize(); }

  /**
   * \returns 4x4 matrix representation of instance
   */
  SOPHUS_FUNC Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.setIdentity();
    homogenious_matrix.block(0, 0, 3, 3) = rotationMatrix();
    homogenious_matrix.col(3).head(3) = translation();
    return homogenious_matrix;
  }

  /**
   * \returns 3x4 matrix representation of instance
   *
   * It returns the three first row of matrix().
   */
  SOPHUS_FUNC Eigen::Matrix<Scalar, 3, 4> matrix3x4() const {
    Eigen::Matrix<Scalar, 3, 4> matrix;
    matrix.block(0, 0, 3, 3) = rotationMatrix();
    matrix.col(3) = translation();
    return matrix;
  }

  /**
   * \brief Assignment operator
   */
  template <typename OtherDerived>
  SOPHUS_FUNC SE3GroupBase<Derived>& operator=(
      const SE3GroupBase<OtherDerived>& other) {
    so3() = other.so3();
    translation() = other.translation();
    return *this;
  }

  /**
   * \brief Group multiplication
   * \see operator*=()
   */
  SOPHUS_FUNC SE3Group<Scalar> operator*(const SE3Group<Scalar>& other) const {
    SE3Group<Scalar> result(*this);
    result *= other;
    return result;
  }

  /**
   * \brief Group action on \f$ \mathbf{R}^3 \f$
   *
   * \param p point \f$p \in \mathbf{R}^3 \f$
   * \returns point \f$p' \in \mathbf{R}^3 \f$,
   *          rotated and translated version of \f$p\f$
   *
   * This function rotates and translates point \f$ p \f$
   * in \f$ \mathbf{R}^3 \f$ by the SE3 transformation \f$R,t\f$
   * (=rotation matrix, translation vector): \f$ p' = R\cdot p + t \f$.
   */
  SOPHUS_FUNC Point operator*(const Point& p) const {
    return so3() * p + translation();
  }

  /**
   * \brief In-place group multiplication
   *
   * \see operator*()
   */
  SOPHUS_FUNC SE3GroupBase<Derived>& operator*=(const SE3Group<Scalar>& other) {
    translation() += so3() * (other.translation());
    so3() *= other.so3();
    return *this;
  }

  /**
   * \returns Rotation matrix
   *
   * deprecated: use rotationMatrix() instead.
   */
  typedef Transformation M3_marcos_dont_like_commas;
  SOPHUS_FUNC
  EIGEN_DEPRECATED const M3_marcos_dont_like_commas rotation_matrix() const {
    return so3().matrix();
  }

  /**
   * \returns Rotation matrix
   */
  SOPHUS_FUNC Eigen::Matrix<Scalar, 3, 3> rotationMatrix() const {
    return so3().matrix();
  }

  /**
   * \brief Mutator of SO3 group
   */
  SOPHUS_FUNC SO3Reference so3() { return static_cast<Derived*>(this)->so3(); }

  /**
   * \brief Accessor of SO3 group
   */
  SOPHUS_FUNC ConstSO3Reference so3() const {
    return static_cast<const Derived*>(this)->so3();
  }

  /**
   * \brief Setter using Affine3
   *
   * \param affine3
   * \pre   3x3 sub-matrix needs to be orthogonal with determinant of 1
   */
  SOPHUS_FUNC void setAffine3(
      const Eigen::Transform<Scalar, 3, Eigen::Affine>& affine3) {
    so3().setRotationMatrix(affine3.matrix().template topLeftCorner<3, 3>());
    translation() = affine3.matrix().template topRightCorner<3, 1>();
  }

  /**
   * \brief Setter of internal unit quaternion representation
   *
   * \param quaternion
   * \pre   the quaternion must not be zero
   *
   * The quaternion is normalized to unit length.
   */
  SOPHUS_FUNC void setQuaternion(const Eigen::Quaternion<Scalar>& quat) {
    so3().setQuaternion(quat);
  }

  /**
   * \brief Setter of unit quaternion using rotation matrix
   *
   * \param rotation_matrix a 3x3 rotation matrix
   * \pre   the 3x3 matrix should be orthogonal and have a determinant of 1
   */
  SOPHUS_FUNC void setRotationMatrix(
      const Eigen::Matrix<Scalar, 3, 3>& rotation_matrix) {
    so3().setQuaternion(Eigen::Quaternion<Scalar>(rotation_matrix));
  }

  /**
   * \brief Mutator of translation vector
   */
  SOPHUS_FUNC TranslationReference translation() {
    return static_cast<Derived*>(this)->translation();
  }

  /**
   * \brief Accessor of translation vector
   */
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return static_cast<const Derived*>(this)->translation();
  }

  /**
   * \brief Accessor of unit quaternion
   *
   * No direct write access is given to ensure the quaternion stays normalized.
   */
  SOPHUS_FUNC typename Eigen::internal::traits<
      Derived>::SO3Type::ConstQuaternionReference
  unit_quaternion() const {
    return so3().unit_quaternion();
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  /**
   * \param   b 6-vector representation of Lie algebra element
   * \returns   derivative of Lie bracket
   *
   * This function returns \f$ \frac{\partial}{\partial a} [a, b]_{se3} \f$
   * with \f$ [a, b]_{se3} \f$ being the lieBracket() of the Lie algebra se3.
   *
   * \see lieBracket()
   */
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

  /**
   * \brief Group exponential
   *
   * \param a tangent space element (6-vector)
   * \returns corresponding element of the group SE3
   *
   * The first three components of \f$ a \f$ represent the translational
   * part \f$ \upsilon \f$ in the tangent space of SE3, while the last three
   * components of \f$ a \f$ represents the rotation vector \f$ \omega \f$.
   *
   * To be more specific, this function computes \f$ \exp(\widehat{a}) \f$
   * with \f$ \exp(\cdot) \f$ being the matrix exponential
   * and \f$ \widehat{\cdot} \f$ the hat()-operator of SE3.
   *
   * \see hat()
   * \see log()
   */
  SOPHUS_FUNC static SE3Group<Scalar> exp(const Tangent& a) {
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

  /**
   * \brief Generators
   *
   * \pre \f$ i \in \{0,1,2,3,4,5\} \f$
   * \returns \f$ i \f$th generator \f$ G_i \f$ of SE3
   *
   * The infinitesimal generators of SE3 are: \f[
   *        G_0 = \left( \begin{array}{cccc}
   *                          0&  0&  0&  1\\
   *                          0&  0&  0&  0\\
   *                          0&  0&  0&  0\\
   *                          0&  0&  0&  0\\
   *                     \end{array} \right),
   *        G_1 = \left( \begin{array}{cccc}
   *                          0&  0&  0&  0\\
   *                          0&  0&  0&  1\\
   *                          0&  0&  0&  0\\
   *                          0&  0&  0&  0\\
   *                     \end{array} \right),
   *        G_2 = \left( \begin{array}{cccc}
   *                          0&  0&  0&  0\\
   *                          0&  0&  0&  0\\
   *                          0&  0&  0&  1\\
   *                          0&  0&  0&  0\\
   *                     \end{array} \right).
   *        G_3 = \left( \begin{array}{cccc}
   *                          0&  0&  0&  0\\
   *                          0&  0& -1&  0\\
   *                          0&  1&  0&  0\\
   *                          0&  0&  0&  0\\
   *                     \end{array} \right),
   *        G_4 = \left( \begin{array}{cccc}
   *                          0&  0&  1&  0\\
   *                          0&  0&  0&  0\\
   *                         -1&  0&  0&  0\\
   *                          0&  0&  0&  0\\
   *                     \end{array} \right),
   *        G_5 = \left( \begin{array}{cccc}
   *                          0& -1&  0&  0\\
   *                          1&  0&  0&  0\\
   *                          0&  0&  0&  0\\
   *                          0&  0&  0&  0\\
   *                     \end{array} \right).
   * \f]
   * \see hat()
   */
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 && i <= 5, "i should be in range [0,5].");
    Tangent e;
    e.setZero();
    e[i] = static_cast<Scalar>(1);
    return hat(e);
  }

  /**
   * \brief ith generator of internal data representation
   *
   * The internal representation is the semi-direct product of SU(2)
   * (unit quaternions) by the 3-dim. Euclidean space (translations).
   */
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

  /**
   * \brief hat-operator
   *
   * \param omega 6-vector representation of Lie algebra element
   * \returns     4x4-matrix representatin of Lie algebra element
   *
   * Formally, the hat-operator of SE3 is defined
   * as \f$ \widehat{\cdot}: \mathbf{R}^6 \rightarrow \mathbf{R}^{4\times 4},
   * \quad \widehat{\omega} = \sum_{i=0}^5 G_i \omega_i \f$
   * with \f$ G_i \f$ being the ith infinitesial generator().
   *
   * \see generator()
   * \see vee()
   */
  SOPHUS_FUNC static Transformation hat(const Tangent& v) {
    Transformation Omega;
    Omega.setZero();
    Omega.template topLeftCorner<3, 3>() =
        SO3Group<Scalar>::hat(v.template tail<3>());
    Omega.col(3).template head<3>() = v.template head<3>();
    return Omega;
  }

  /**
   * \brief Lie bracket
   *
   * \param a 6-vector representation of Lie algebra element
   * \param b 6-vector representation of Lie algebra element
   * \returns      6-vector representation of Lie algebra element
   *
   * It computes the bracket of SE3. To be more specific, it
   * computes \f$ [a, b]_{se3}
   * := [\widehat{a}, \widehat{b}]^\vee \f$
   * with \f$ [A,B] = AB-BA \f$ being the matrix
   * commutator, \f$ \widehat{\cdot} \f$ the
   * hat()-operator and \f$ (\cdot)^\vee \f$ the vee()-operator of SE3.
   *
   * \see hat()
   * \see vee()
   */
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

  /**
   * \brief Logarithmic map
   *
   * \param other element of the group SE3
   * \returns     corresponding tangent space element
   *              (translational part \f$ \upsilon \f$
   *               and rotation vector \f$ \omega \f$)
   *
   * Computes the logarithmic, the inverse of the group exponential.
   * To be specific, this function computes \f$ \log({\cdot})^\vee \f$
   * with \f$ \vee(\cdot) \f$ being the matrix logarithm
   * and \f$ \vee{\cdot} \f$ the vee()-operator of SE3.
   *
   * \see exp()
   * \see vee()
   */
  SOPHUS_FUNC static Tangent log(const SE3Group<Scalar>& se3) {
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

  /**
   * \brief vee-operator
   *
   * \param Omega 4x4-matrix representation of Lie algebra element
   * \returns     6-vector representatin of Lie algebra element
   *
   * This is the inverse of the hat()-operator.
   *
   * \see hat()
   */
  SOPHUS_FUNC static Tangent vee(const Transformation& Omega) {
    Tangent upsilon_omega;
    upsilon_omega.template head<3>() = Omega.col(3).template head<3>();
    upsilon_omega.template tail<3>() =
        SO3Group<Scalar>::vee(Omega.template topLeftCorner<3, 3>());
    return upsilon_omega;
  }
};

/**
 * \brief SE3 default type - Constructors and default storage for SE3 Type
 */
template <typename _Scalar, int _Options>
class SE3Group : public SE3GroupBase<SE3Group<_Scalar, _Options>> {
  typedef SE3GroupBase<SE3Group<_Scalar, _Options>> Base;

 public:
  /** \brief scalar type */
  typedef typename Eigen::internal::traits<SE3Group<_Scalar, _Options>>::Scalar
      Scalar;
  /** \brief SO3 reference type */
  typedef
      typename Eigen::internal::traits<SE3Group<_Scalar, _Options>>::SO3Type&
          SO3Reference;
  /** \brief SO3 const reference type */
  typedef const typename Eigen::internal::traits<
      SE3Group<_Scalar, _Options>>::SO3Type& ConstSO3Reference;
  /** \brief translation reference type */
  typedef typename Eigen::internal::traits<
      SE3Group<_Scalar, _Options>>::TranslationType& TranslationReference;
  /** \brief translation const reference type */
  typedef const typename Eigen::internal::traits<
      SE3Group<_Scalar, _Options>>::TranslationType& ConstTranslationReference;

  /** \brief group transfomation type */
  typedef typename Base::Transformation Transformation;
  /** \brief point type */
  typedef typename Base::Point Point;
  /** \brief tangent vector type */
  typedef typename Base::Tangent Tangent;
  /** \brief adjoint transformation type */
  typedef typename Base::Adjoint Adjoint;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * \brief Default constructor
   *
   * Initialize Eigen::Quaternion to identity rotation and translation to zero.
   */
  SOPHUS_FUNC SE3Group() : translation_(Eigen::Matrix<Scalar, 3, 1>::Zero()) {}

  /**
   * \brief Copy constructor
   */
  template <typename OtherDerived>
  SOPHUS_FUNC SE3Group(const SE3GroupBase<OtherDerived>& other)
      : so3_(other.so3()), translation_(other.translation()) {}

  /**
   * \brief Constructor from SO3 and translation vector
   */
  template <typename OtherDerived>
  SOPHUS_FUNC SE3Group(const SO3GroupBase<OtherDerived>& so3,
                       const Point& translation)
      : so3_(so3), translation_(translation) {}

  /**
   * \brief Constructor from rotation matrix and translation vector
   *
   * \pre rotation matrix need to be orthogonal with determinant of 1
   */
  SOPHUS_FUNC
  SE3Group(const Eigen::Matrix<Scalar, 3, 3>& rotation_matrix,
           const Point& translation)
      : so3_(rotation_matrix), translation_(translation) {}

  /**
   * \brief Constructor from quaternion and translation vector
   *
   * \pre quaternion must not be zero
   */
  SOPHUS_FUNC SE3Group(const Eigen::Quaternion<Scalar>& quaternion,
                       const Point& translation)
      : so3_(quaternion), translation_(translation) {}

  /**
   * \brief Constructor from 4x4 matrix
   *
   * \pre top-left 3x3 sub-matrix need to be orthogonal with determinant of 1
   */
  SOPHUS_FUNC explicit SE3Group(const Eigen::Matrix<Scalar, 4, 4>& T)
      : so3_(T.template topLeftCorner<3, 3>()),
        translation_(T.template block<3, 1>(0, 3)) {}

  /**
   * \brief Constructor from Affine3
   *
   * \pre top-left 3x3 sub-matrix need to be orthogonal with determinant of 1
   */
  SOPHUS_FUNC explicit SE3Group(
      const Eigen::Transform<Scalar, 3, Eigen::Affine>& affine3)
      : so3_(affine3.matrix().template topLeftCorner<3, 3>()),
        translation_(affine3.matrix().template block<3, 1>(0, 3)) {}

  /**
   * \returns pointer to internal data
   *
   * This provides unsafe read/write access to internal data. SE3 is represented
   * by a pair of an SO3 element (4 parameters) and translation vector (three
   * parameters). The user needs to take care of that the quaternion
   * stays normalized.
   *
   * Note: The first three Scalars represent the imaginary parts, while the
   * forth Scalar represent the real part.
   *
   * /see normalize()
   */
  SOPHUS_FUNC Scalar* data() {
    // so3_ and translation_ are layed out sequentially with no padding
    return so3_.data();
  }

  /**
   * \returns const pointer to internal data
   *
   * Const version of data().
   */
  SOPHUS_FUNC const Scalar* data() const {
    // so3_ and translation_ are layed out sequentially with no padding
    return so3_.data();
  }

  /**
   * \brief Accessor of SO3
   */
  SOPHUS_FUNC SO3Reference so3() { return so3_; }

  /**
   * \brief Mutator of SO3
   */
  SOPHUS_FUNC ConstSO3Reference so3() const { return so3_; }

  /**
   * \brief Mutator of translation vector
   */
  SOPHUS_FUNC TranslationReference translation() { return translation_; }

  /**
   * \brief Accessor of translation vector
   */
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return translation_;
  }

 protected:
  Sophus::SO3Group<Scalar> so3_;
  Eigen::Matrix<Scalar, 3, 1> translation_;
};

}  // end namespace

namespace Eigen {
/**
 * \brief Specialisation of Eigen::Map for SE3GroupBase
 *
 * Allows us to wrap SE3 Objects around POD array
 * (e.g. external c style quaternion)
 */
template <typename _Scalar, int _Options>
class Map<Sophus::SE3Group<_Scalar>, _Options>
    : public Sophus::SE3GroupBase<Map<Sophus::SE3Group<_Scalar>, _Options>> {
  typedef Sophus::SE3GroupBase<Map<Sophus::SE3Group<_Scalar>, _Options>> Base;

 public:
  /** \brief scalar type */
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  /** \brief translation reference type */
  typedef typename Eigen::internal::traits<Map>::TranslationType&
      TranslationReference;
  /** \brief translation const reference type */
  typedef const typename Eigen::internal::traits<Map>::TranslationType&
      ConstTranslationReference;
  /** \brief SO3 reference type */
  typedef typename Eigen::internal::traits<Map>::SO3Type& SO3Reference;
  /** \brief SO3 const reference type */
  typedef const typename Eigen::internal::traits<Map>::SO3Type&
      ConstSO3Reference;

  /** \brief group transfomation type */
  typedef typename Base::Transformation Transformation;
  /** \brief point type */
  typedef typename Base::Point Point;
  /** \brief tangent vector type */
  typedef typename Base::Tangent Tangent;
  /** \brief adjoint transformation type */
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar* coeffs)
      : so3_(coeffs),
        translation_(coeffs + Sophus::SO3Group<Scalar>::num_parameters) {}

  /**
   * \brief Mutator of SO3
   */
  SOPHUS_FUNC SO3Reference so3() { return so3_; }

  /**
   * \brief Accessor of SO3
   */
  SOPHUS_FUNC ConstSO3Reference so3() const { return so3_; }

  /**
   * \brief Mutator of translation vector
   */
  SOPHUS_FUNC TranslationReference translation() { return translation_; }

  /**
   * \brief Accessor of translation vector
   */
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return translation_;
  }

 protected:
  Map<Sophus::SO3Group<Scalar>, _Options> so3_;
  Map<Eigen::Matrix<Scalar, 3, 1>, _Options> translation_;
};

/**
 * \brief Specialisation of Eigen::Map for const SE3GroupBase
 *
 * Allows us to wrap SE3 Objects around POD array
 * (e.g. external c style quaternion)
 */
template <typename _Scalar, int _Options>
class Map<const Sophus::SE3Group<_Scalar>, _Options>
    : public Sophus::SE3GroupBase<
          Map<const Sophus::SE3Group<_Scalar>, _Options>> {
  typedef Sophus::SE3GroupBase<Map<const Sophus::SE3Group<_Scalar>, _Options>>
      Base;

 public:
  /** \brief scalar type */
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  /** \brief translation const reference type */
  typedef const typename Eigen::internal::traits<Map>::TranslationType&
      ConstTranslationReference;
  /** \brief SO3 const reference type */
  typedef const typename Eigen::internal::traits<Map>::SO3Type&
      ConstSO3Reference;

  /** \brief group transfomation type */
  typedef typename Base::Transformation Transformation;
  /** \brief point type */
  typedef typename Base::Point Point;
  /** \brief tangent vector type */
  typedef typename Base::Tangent Tangent;
  /** \brief adjoint transformation type */
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(const Scalar* coeffs)
      : so3_(coeffs),
        translation_(coeffs + Sophus::SO3Group<Scalar>::num_parameters) {}

  SOPHUS_FUNC Map(const Scalar* trans_coeffs, const Scalar* rot_coeffs)
      : so3_(rot_coeffs), translation_(trans_coeffs) {}

  /**
   * \brief Accessor of SO3
   */
  SOPHUS_FUNC ConstSO3Reference so3() const { return so3_; }

  /**
   * \brief Accessor of translation vector
   */
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return translation_;
  }

 protected:
  const Map<const Sophus::SO3Group<Scalar>, _Options> so3_;
  const Map<const Eigen::Matrix<Scalar, 3, 1>, _Options> translation_;
};
}

#endif
