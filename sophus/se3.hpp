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
template<typename _Scalar, int _Options=0> class SE3Group;
typedef SOPHUS_DEPRECATED SE3Group<double> SE3;
typedef SE3Group<double> SE3d; /**< double precision SE3 */
typedef SE3Group<float> SE3f;  /**< single precision SE3 */
typedef Matrix<double,6,1> Vector6d;
typedef Matrix<double,6,6> Matrix6d;
typedef Matrix<float,6,1> Vector6f;
typedef Matrix<float,6,6> Matrix6f;
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

/**
 * \brief SE3 base type - implements SE3 class but is storage agnostic
 *
 * [add more detailed description/tutorial]
 */
template<typename Derived>
class SE3GroupBase {
public:
  typedef typename internal::traits<Derived>::Scalar Scalar;
  typedef typename internal::traits<Derived>::TranslationType TranslationType;
  typedef typename internal::traits<Derived>::SO3Type SO3Type;
  /** \brief degree of freedom of group */
  static const int DoF = 6;
  /** \brief number of internal parameters used */
  static const int num_parameters = 7;

  /**
   * \brief Adjoint transformation
   *
   * This function return the adjoint transformation \f$ Ad \f$ of the
   * group instance \f$ A \f$  such that for all \f$ x \f$
   * it holds that \f$ \widehat{Ad_A\cdot x} = A\widehat{x}A^{-1} \f$
   * with \f$\ \widehat{\cdot} \f$ being the hat()-operator.
   */
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

  /**
   * \returns copy of instance casted to NewScalarType
   */
  template<typename NewScalarType>
  inline SE3Group<NewScalarType> cast() const {
    return
        SE3Group<NewScalarType>(so3().template cast<NewScalarType>(),
                                translation().template cast<NewScalarType>() );
  }

  /**
   * \brief Fast group multiplication
   *
   * This method is a fast version of operator*=(), since it does not perform
   * normalization. It is up to the user to call normalize() once in a while.
   *
   * \see operator*=()
   */
  inline
  void fastMultiply(const SE3Group<Scalar>& other) {
    translation() += so3()*(other.translation());
    so3().fastMultiply(other.so3());
  }

  /**
   * \returns Group inverse of instance
   */
  inline
  const SE3Group<Scalar> inverse() const {
    const SO3Group<Scalar> invR = so3().inverse();
    return SE3Group<Scalar>(invR, invR*(translation()
                                        *static_cast<Scalar>(-1) ) );
  }

  /**
   * \brief Logarithmic map
   *
   * \returns tangent space representation
   *          (translational part and rotation vector) of instance
   *
   * \see  log().
   */
  inline
  const Matrix<Scalar,6,1> log() const {
    return log(*this);
  }

  /**
   * \brief Normalize SO3 element
   *
   * It re-normalizes the SO3 element. This method only needs to
   * be called in conjunction with fastMultiply() or data() write access.
   */
  inline
  void normalize() {
    so3().normalize();
  }

  /**
   * \returns 4x4 matrix representation of instance
   */
  inline
  const Matrix<Scalar,4,4> matrix() const {
    Matrix<Scalar,4,4> homogenious_matrix;
    homogenious_matrix.setIdentity();
    homogenious_matrix.block(0,0,3,3) = rotation_matrix();
    homogenious_matrix.col(3).head(3) = translation();
    return homogenious_matrix;
  }

  /**
   * \returns 3x4 matrix representation of instance
   *
   * It returns the three first row of matrix().
   */
  inline
  const Matrix<Scalar,3,4> matrix3x4() const {
    Matrix<Scalar,3,4> matrix;
    matrix.block(0,0,3,3) = rotation_matrix();
    matrix.col(3) = translation();
    return matrix;
  }

  /**
   * \brief Assignment operator
   */
  template<typename OtherDerived> inline
  SE3GroupBase<Derived>& operator= (const SE3GroupBase<OtherDerived> & other) {
    so3() = other.so3();
    translation() = other.translation();
    return *this;
  }

  /**
   * \brief Group multiplication
   * \see operator*=()
   */
  inline
  const SE3Group<Scalar> operator*(const SE3Group<Scalar>& other) const {
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
   * This function rotates aand translates point \f$ p \f$
   * in \f$ \mathbf{R}^3 \f$ by the SE3 transformation \f$R,t\f$
   * (=rotation matrix, translation vector): \f$ p' = R\cdot p + t \f$.
   */
  inline
  const Matrix<Scalar,3,1> operator*(const Matrix<Scalar,3,1> & p) const {
    return so3()*p + translation();
  }

  /**
   * \brief In-place group multiplication
   *
   * \see fastMultiply()
   * \see operator*()
   */
  inline
  void operator*=(const SE3Group<Scalar>& other) {
    fastMultiply(other);
    normalize();
  }


  /**
   * \returns Rotation matrix
   */
  inline
  const Matrix<Scalar,3,3> rotation_matrix() const {
    return so3().matrix();
  }

  /**
   * \brief Read/write access to SO3 group
   */
  EIGEN_STRONG_INLINE
  SO3Type& so3() {
      return static_cast<Derived*>(this)->so3();
  }

  /**
   * \brief Read access to SO3 group
   */
  EIGEN_STRONG_INLINE
  const SO3Type& so3() const {
      return static_cast<const Derived*>(this)->so3();
  }

  /**
   * \brief Setter of internal unit quaternion representation
   *
   * \param quaternion
   * \pre   the quaternion must not be zero
   *
   * The quaternion is normalized to unit length.
   */
  inline
  void setQuaternion(const typename SO3Type::QuaternionType& quat) {
    return so3().setQuaternion(quat);
  }

  /**
   * \brief Setter of unit quaternion using rotation matrix
   *
   * \param rotation_matrix a 3x3 rotation matrix
   * \pre   the 3x3 matrix should be orthogonal and have a determinant of 1
   */
  inline
  void setRotationMatrix(const Matrix3d & rotation_matrix) {
    so3().setQuaternion(SO3Type::QuaternionType(rotation_matrix));
  }

  /**
   * \brief Read/write access to translation vector
   */
  EIGEN_STRONG_INLINE
  TranslationType& translation() {
      return static_cast<Derived*>(this)->translation();
  }

  /**
   * \brief Read access to translation vector
   */
  EIGEN_STRONG_INLINE
  const TranslationType& translation() const {
      return static_cast<const Derived*>(this)->translation();
  }

  /**
   * \brief Read access to unit quaternion
   *
   * No direct write access is given to ensure the quaternion stays normalized.
   */
  inline
  const typename SO3Type::QuaternionType& unit_quaternion() const {
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
  inline static
  const SE3Group<Scalar> exp(const Matrix<Scalar,6,1> & a) {
    Matrix<Scalar,3,1> upsilon = a.template head<3>();
    Matrix<Scalar,3,1> omega = a.template tail<3>();

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
  inline static
  const Matrix<Scalar,4,4> generator(int i) {
    assert(i>=0 && i<6);
    Matrix<Scalar,6,1> e;
    e.setZero();
    e[i] = 1.f;
    return hat(e);
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
  inline static
  const Matrix<Scalar,4,4> hat(const Matrix<Scalar,6,1> & v) {
    Matrix<Scalar,4,4> Omega;
    Omega.setZero();
    Omega.template topLeftCorner<3,3>()
        = SO3Group<Scalar>::hat(v.template tail<3>());
    Omega.col(3).template head<3>() = v.template head<3>();
    return Omega;
  }

  /**
   * \brief Lie bracket
   *
   * \param omega1 6-vector representation of Lie algebra element
   * \param omega2 6-vector representation of Lie algebra element
   * \returns      6-vector representation of Lie algebra element
   *
   * It computes the bracket of SE3. To be more specific, it
   * computes \f$ [\omega_1, \omega_2]_{se3}
   * := [\widehat{\omega_1}, \widehat{\omega_2}]^\vee \f$
   * with \f$ [A,B] = AB-BA \f$ being the matrix
   * commutator, \f$ \widehat{\cdot} \f$ the
   * hat()-operator and \f$ (\cdot)^\vee \f$ the vee()-operator of SE3.
   *
   * \see hat()
   * \see vee()
   */
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
  inline static
  const Matrix<Scalar,6,1> vee(const Matrix<Scalar,4,4> & Omega) {
    Matrix<Scalar,6,1> upsilon_omega;
    upsilon_omega.template head<3>() = Omega.col(3).template head<3>();
    upsilon_omega.template tail<3>()
        = SO3Group<Scalar>::vee(Omega.template topLeftCorner<3,3>());
    return upsilon_omega;
  }
};

/**
 * \brief SE3 default type - Constructors and default storage for SE3 Type
 */
template<typename _Scalar, int _Options>
class SE3Group : public SE3GroupBase<SE3Group<_Scalar,_Options> > {
public:
  typedef typename internal::traits<SE3Group<_Scalar,_Options> >
  ::Scalar Scalar;
  typedef typename internal::traits<SE3Group<_Scalar,_Options> >
  ::TranslationType TranslationType;
  typedef typename internal::traits<SE3Group<_Scalar,_Options> >
  ::SO3Type SO3Type;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * \brief Default constructor
   *
   * Initialize Quaternion to identity rotation and translation to zero.
   */
  inline
  SE3Group()
      : translation_( TranslationType::Zero() )
  {
  }

  /**
   * \brief Copy constructor
   */
  template<typename OtherDerived> inline
  SE3Group(const SE3GroupBase<OtherDerived> & other)
    : so3_(other.so3()), translation_(other.translation()) {
  }

  /**
   * \brief Constructor from SO3 and translation vector
   */
  template<typename OtherDerived> inline
  SE3Group(const SO3GroupBase<OtherDerived> & so3,
           const Matrix<Scalar,3,1> & translation)
    : so3_(so3), translation_(translation) {
  }

  /**
   * \brief Constructor from rotation matrix and translation vector
   *
   * \pre rotation matrix need to be orthogonal with determinant of 1
   */
  inline
  SE3Group(const Matrix3d & rotation_matrix,
           const Matrix<Scalar,3,1> & translation)
    : so3_(rotation_matrix), translation_(translation) {
  }

  /**
   * \brief Constructor from quaternion and translation vector
   *
   * \pre quaternion must not be zero
   */
  inline
  SE3Group(const Quaternion<Scalar> & quaternion,
           const Matrix<Scalar,3,1> & translation)
    : so3_(quaternion), translation_(translation) {
  }

  /**
   * \brief Constructor from 4x4 matrix
   *
   * \pre 3x3 sub-matrix need to be orthogonal with determinant of 1
   */
  inline
  SE3Group(const Eigen::Matrix<Scalar,4,4>& T)
    : so3_(T.template topLeftCorner<3,3>()),
      translation_(T.template block<3,1>(0,3)) {
  }

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
  EIGEN_STRONG_INLINE
  Scalar* data() {
    // so3_ and translation_ are layed out sequentially with no padding
    return so3_.data();
  }

  /**
   * \returns const pointer to internal data
   *
   * Const version of data().
   */
  EIGEN_STRONG_INLINE
  const Scalar* data() const {
      // so3_ and translation_ are layed out sequentially with no padding
      return so3_.data();
  }

  /**
   * \brief Read access to SO3
   */
  EIGEN_STRONG_INLINE
  SO3Type& so3() {
    return so3_;
  }

  /**
   * \brief Read/write access to SO3
   */
  EIGEN_STRONG_INLINE
  const SO3Type& so3() const {
    return so3_;
  }

  /**
   * \brief Read/write access to translation vector
   */
  EIGEN_STRONG_INLINE
  TranslationType& translation() {
    return translation_;
  }

  /**
   * \brief Read access to translation vector
   */
  EIGEN_STRONG_INLINE
  const TranslationType& translation() const {
    return translation_;
  }

protected:
  SO3Type so3_;
  TranslationType translation_;
};


} // end namespace


namespace Eigen {
/**
 * \brief Specialisation of Eigen::Map for SE3GroupBase
 *
 * Allows us to wrap SE3 Objects around POD array
 * (e.g. external c style quaternion)
 */
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
  Map(Scalar* coeffs) : so3_(coeffs), translation_(coeffs+4) {
  }

  /**
   * \brief Read/write access to SO3
   */
  EIGEN_STRONG_INLINE
  SO3Type& so3() {
    return so3_;
  }

  /**
   * \brief Read access to SO3
   */
  EIGEN_STRONG_INLINE
  const SO3Type& so3() const {
    return so3_;
  }

  /**
   * \brief Read/write access to translation vector
   */
  EIGEN_STRONG_INLINE
  TranslationType& translation() {
    return translation_;
  }

  /**
   * \brief Read access to translation vector
   */
  EIGEN_STRONG_INLINE
  const TranslationType& translation() const {
    return translation_;
  }

protected:
  SO3Type so3_;
  TranslationType translation_;
};

/**
 * \brief Specialisation of Eigen::Map for const SE3GroupBase
 *
 * Allows us to wrap SE3 Objects around POD array
 * (e.g. external c style quaternion)
 */
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
    : so3_(coeffs), translation_(coeffs+4) {
  }

  EIGEN_STRONG_INLINE
  Map(const Scalar* trans_coeffs, const Scalar* rot_coeffs)
    : translation_(trans_coeffs), so3_(rot_coeffs){
  }

  /**
   * \brief Read access to SO3
   */
  EIGEN_STRONG_INLINE
  const SO3Type& so3() const {
    return so3_;
  }

  /**
   * \brief Read access to translation vector
   */
  EIGEN_STRONG_INLINE
  const TranslationType& translation() const {
    return translation_;
  }

protected:
  const SO3Type so3_;
  const TranslationType translation_;
};

}

#endif
