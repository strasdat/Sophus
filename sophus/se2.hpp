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

////////////////////////////////////////////////////////////////////////////
// Forward Declarations / typedefs
////////////////////////////////////////////////////////////////////////////

namespace Sophus {
template<typename _Scalar, int _Options=0> class SE2Group;
typedef SOPHUS_DEPRECATED SE2Group<double> SE2;
typedef SE2Group<double> SE2d; /**< double precision SE2 */
typedef SE2Group<float> SE2f;  /**< single precision SE2 */
}

////////////////////////////////////////////////////////////////////////////
// Eigen Traits (For querying derived types in CRTP hierarchy)
////////////////////////////////////////////////////////////////////////////

namespace Eigen {
namespace internal {

template<typename _Scalar, int _Options>
struct traits<Sophus::SE2Group<_Scalar,_Options> > {
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,2,1> TranslationType;
  typedef Sophus::SO2Group<Scalar> SO2Type;
};

template<typename _Scalar, int _Options>
struct traits<Map<Sophus::SE2Group<_Scalar>, _Options> >
    : traits<Sophus::SE2Group<_Scalar, _Options> > {
  typedef _Scalar Scalar;
  typedef Map<Matrix<Scalar,2,1>,_Options> TranslationType;
  typedef Map<Sophus::SO2Group<Scalar>,_Options> SO2Type;
};

template<typename _Scalar, int _Options>
struct traits<Map<const Sophus::SE2Group<_Scalar>, _Options> >
    : traits<const Sophus::SE2Group<_Scalar, _Options> > {
  typedef _Scalar Scalar;
  typedef Map<const Matrix<Scalar,2,1>,_Options> TranslationType;
  typedef Map<const Sophus::SO2Group<Scalar>,_Options> SO2Type;
};

}
}

namespace Sophus {
using namespace Eigen;
using namespace std;

/**
 * \brief SE2 base type - implements SE2 class but is storage agnostic
 *
 * [add more detailed description/tutorial]
 */
template<typename Derived>
class SE2GroupBase {
public:
  typedef typename internal::traits<Derived>::Scalar Scalar;
  typedef typename internal::traits<Derived>::TranslationType TranslationType;
  typedef typename internal::traits<Derived>::SO2Type SO2Type;
  /** \brief degree of freedom of group */
  static const int DoF = 3;
  /** \brief number of internal parameters used */
  static const int num_parameters = 4;

  /**
   * \brief Adjoint transformation
   *
   * This function return the adjoint transformation \f$ Ad \f$ of the
   * group instance \f$ A \f$  such that for all \f$ x \f$
   * it holds that \f$ \widehat{Ad_A\cdot x} = A\widehat{x}A^{-1} \f$
   * with \f$\ \widehat{\cdot} \f$ being the hat()-operator.
   */
  inline
  const Matrix<Scalar, 3, 3> Adj() const {
    const Matrix<Scalar,2,2> & R = so2().matrix();
    Matrix<Scalar,3,3> res;
    res.setIdentity();
    res.template topLeftCorner<2,2>() = R;
    res(0,2) =  translation()[1];
    res(1,2) = -translation()[0];
    return res;
  }

  /**
   * \returns copy of instance casted to NewScalarType
   */
  template<typename NewScalarType>
  inline SE2Group<NewScalarType> cast() const {
    return
        SE2Group<NewScalarType>(so2().template cast<NewScalarType>(),
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
  void fastMultiply(const SE2Group<Scalar>& other) {
    translation() += so2()*(other.translation());
    so2().fastMultiply(other.so2());
  }

  /**
   * \returns Group inverse of instance
   */
  inline
  const SE2Group<Scalar> inverse() const {
    const SO2Group<Scalar> invR = so2().inverse();
    return SE2Group<Scalar>(invR, invR*(translation()
                                        *static_cast<Scalar>(-1) ) );
  }

  /**
   * \brief Logarithmic map
   *
   * \returns tangent space representation
   *          (translational part and rotation angle) of instance
   *
   * \see  log().
   */
  inline
  const Matrix<Scalar,3,1> log() const {
    return log(*this);
  }

  /**
   * \brief Normalize SO2 element
   *
   * It re-normalizes the SO2 element. This method only needs to
   * be called in conjunction with fastMultiply() or data() write access.
   */
  inline
  void normalize() {
    so2().normalize();
  }

  /**
   * \returns 3x3 matrix representation of instance
   */
  inline
  const Matrix<Scalar,3,3> matrix() const {
    Matrix<Scalar,3,3> homogenious_matrix;
    homogenious_matrix.setIdentity();
    homogenious_matrix.block(0,0,2,2) = rotation_matrix();
    homogenious_matrix.col(2).head(2) = translation();
    return homogenious_matrix;
  }

  /**
   * \returns 2x3 matrix representation of instance
   *
   * It returns the three first row of matrix().
   */
  inline
  const Matrix<Scalar,2,3> matrix2x3() const {
    Matrix<Scalar,2,3> matrix;
    matrix.block(0,0,2,2) = rotation_matrix();
    matrix.col(2) = translation();
    return matrix;
  }

  /**
   * \brief Assignment operator
   */
  template<typename OtherDerived> inline
  SE2GroupBase<Derived>& operator= (const SE2GroupBase<OtherDerived> & other) {
    so2() = other.so2();
    translation() = other.translation();
    return *this;
  }

  /**
   * \brief Group multiplication
   * \see operator*=()
   */
  inline
  const SE2Group<Scalar> operator*(const SE2Group<Scalar>& other) const {
    SE2Group<Scalar> result(*this);
    result *= other;
    return result;
  }

  /**
   * \brief Group action on \f$ \mathbf{R}^2 \f$
   *
   * \param p point \f$p \in \mathbf{R}^2 \f$
   * \returns point \f$p' \in \mathbf{R}^2 \f$,
   *          rotated and translated version of \f$p\f$
   *
   * This function rotates aand translates point \f$ p \f$
   * in \f$ \mathbf{R}^2 \f$ by the SE2 transformation \f$R,t\f$
   * (=rotation matrix, translation vector): \f$ p' = R\cdot p + t \f$.
   */
  inline
  const Matrix<Scalar,2,1> operator*(const Matrix<Scalar,2,1> & p) const {
    return so2()*p + translation();
  }

  /**
   * \brief In-place group multiplication
   *
   * \see fastMultiply()
   * \see operator*()
   */
  inline
  void operator*=(const SE2Group<Scalar>& other) {
    fastMultiply(other);
    normalize();
  }


  /**
   * \returns Rotation matrix
   */
  inline
  const Matrix<Scalar,2,2> rotation_matrix() const {
    return so2().matrix();
  }

  /**
   * \brief Read/write access to SO2 group
   */
  EIGEN_STRONG_INLINE
  SO2Type& so2() {
      return static_cast<Derived*>(this)->so2();
  }

  /**
   * \brief Read access to SO2 group
   */
  EIGEN_STRONG_INLINE
  const SO2Type& so2() const {
      return static_cast<const Derived*>(this)->so2();
  }

  /**
   * \brief Setter of internal unit complex number representation
   *
   * \param complex
   * \pre   the complex number must not be zero
   *
   * The complex number is normalized to unit length.
   */
  inline
  void setComplex(const typename SO2Type::ComplexType& complex) {
    return so2().setComplex(complex);
  }

  /**
   * \brief Setter of unit complex number using rotation matrix
   *
   * \param R a 2x2 matrix
   * \pre     the 2x2 matrix should be orthogonal and have a determinant of 1
   */
  inline
  void setRotationMatrix(const Matrix<Scalar,2,2> & R) {
    so2().setComplex(static_cast<Scalar>(0.5)*(R(0,0)+R(1,1)),
                     static_cast<Scalar>(0.5)*(R(1,0)-R(0,1)));
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
   * \brief Read access to unit complex number
   *
   * No direct write access is given to ensure the complex number stays
   * normalized.
   */
  inline
  const typename SO2Type::ComplexType& unit_complex() const {
    return so2().unit_complex();
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  /**
   * \param   b 3-vector representation of Lie algebra element
   * \returns   derivative of Lie bracket
   *
   * This function returns \f$ \frac{\partial}{\partial a} [a, b]_{se2} \f$
   * with \f$ [a, b]_{se2} \f$ being the lieBracket() of the Lie algebra se2.
   *
   * \see lieBracket()
   */
  inline static
  const Matrix<Scalar,3,3> d_lieBracketab_by_d_a(const Matrix<Scalar,3,1> & b) {
    Matrix<Scalar,2,1> upsilon2 = b.template head<2>();
    double theta2 = b[2];

    Matrix<Scalar,3,3> res;
    res <<      0., theta2, -upsilon2[1]
        ,  -theta2,     0.,  upsilon2[0]
        ,       0.,     0.,           0.;
    return res;
  }

  /**
   * \brief Group exponential
   *
   * \param a tangent space element (3-vector)
   * \returns corresponding element of the group SE2
   *
   * The first two components of \f$ a \f$ represent the translational
   * part \f$ \upsilon \f$ in the tangent space of SE2, while the last
   * components of \f$ a \f$ is the rotation angle \f$ \theta \f$.
   *
   * To be more specific, this function computes \f$ \exp(\widehat{a}) \f$
   * with \f$ \exp(\cdot) \f$ being the matrix exponential
   * and \f$ \widehat{\cdot} \f$ the hat()-operator of SE2.
   *
   * \see hat()
   * \see log()
   */
  inline static
  const SE2Group<Scalar> exp(const Matrix<Scalar,3,1> & a) {
    Matrix<Scalar,2,1> upsilon = a.template head<2>();
    Scalar theta = a[2];
    SO2Type so2 = SO2Type::exp(theta);
    Scalar sin_theta_by_theta;
    Scalar one_minus_cos_theta_by_theta;

    if(abs(theta)<SophusConstants<Scalar>::epsilon()) {
      Scalar theta_sq = theta*theta;
      sin_theta_by_theta
          = static_cast<Scalar>(1.) - static_cast<Scalar>(1./6.)*theta_sq;
      one_minus_cos_theta_by_theta
          = static_cast<Scalar>(0.5)*theta
            - static_cast<Scalar>(1./24.)*theta*theta_sq;
    } else {
      sin_theta_by_theta = so2.unit_complex().y()/theta;
      one_minus_cos_theta_by_theta
          = (static_cast<Scalar>(1.) - so2.unit_complex().x())/theta;
    }
    Matrix<Scalar,2,2> V;
    V(0,0) = sin_theta_by_theta; V(0,1) = -one_minus_cos_theta_by_theta;
    V(1,0) = one_minus_cos_theta_by_theta; V(1,1) = sin_theta_by_theta;
    return SE2Group<Scalar>(so2,V*upsilon);
  }

  /**
   * \brief Generators
   *
   * \pre \f$ i \in \{0,1,2\} \f$
   * \returns \f$ i \f$th generator \f$ G_i \f$ of SE2
   *
   * The infinitesimal generators of SE2 are: \f[
   *        G_0 = \left( \begin{array}{ccc}
   *                          0&  0&  1\\
   *                          0&  0&  0\\
   *                          0&  0&  0\\
   *                     \end{array} \right),
   *        G_1 = \left( \begin{array}{cccc}
   *                          0&  0&  0\\
   *                          0&  0&  1\\
   *                          0&  0&  0\\
   *                     \end{array} \right),
   *        G_2 = \left( \begin{array}{cccc}
   *                          0&  0&  0&\\
   *                          0&  0& -1&\\
   *                          0&  1&  0&\\
   *                     \end{array} \right),
   * \f]
   * \see hat()
   */
  inline static
  const Matrix<Scalar,3,3> generator(int i) {
    assert(i>=0 && i<3);
    Matrix<Scalar,3,1> e;
    e.setZero();
    e[i] = 1.f;
    return hat(e);
  }

  /**
   * \brief hat-operator
   *
   * \param omega 3-vector representation of Lie algebra element
   * \returns     3x3-matrix representatin of Lie algebra element
   *
   * Formally, the hat-operator of SE2 is defined
   * as \f$ \widehat{\cdot}: \mathbf{R}^3 \rightarrow \mathbf{R}^{2\times 2},
   * \quad \widehat{\omega} = \sum_{i=0}^2 G_i \omega_i \f$
   * with \f$ G_i \f$ being the ith infinitesial generator().
   *
   * \see generator()
   * \see vee()
   */
  inline static
  const Matrix<Scalar,3,3> hat(const Matrix<Scalar,3,1> & v) {
    Matrix<Scalar,3,3> Omega;
    Omega.setZero();
    Omega.template topLeftCorner<2,2>() = SO2Group<Scalar>::hat(v[2]);
    Omega.col(2).template head<2>() = v.template head<2>();
    return Omega;
  }

  /**
   * \brief Lie bracket
   *
   * \param a 3-vector representation of Lie algebra element
   * \param b 3-vector representation of Lie algebra element
   * \returns 3-vector representation of Lie algebra element
   *
   * It computes the bracket of SE2. To be more specific, it
   * computes \f$ [a, b]_{se2}
   * := [\widehat{a_1}, \widehat{b_2}]^\vee \f$
   * with \f$ [A,B] = AB-BA \f$ being the matrix
   * commutator, \f$ \widehat{\cdot} \f$ the
   * hat()-operator and \f$ (\cdot)^\vee \f$ the vee()-operator of SE2.
   *
   * \see hat()
   * \see vee()
   */
  inline static
  const Matrix<Scalar,3,1> lieBracket(const Matrix<Scalar,3,1> & a,
                                      const Matrix<Scalar,3,1> & b) {
    Matrix<Scalar,2,1> upsilon1 = a.template head<2>();
    Matrix<Scalar,2,1> upsilon2 = b.template head<2>();
    Scalar theta1 = a[2];
    Scalar theta2 = b[2];

    return Matrix<Scalar,3,1>(-theta1*upsilon2[1] + theta2*upsilon1[1],
                              theta1*upsilon2[0] - theta2*upsilon1[0],
                              static_cast<Scalar>(0));
  }

  /**
   * \brief Logarithmic map
   *
   * \param other element of the group SE2
   * \returns     corresponding tangent space element
   *              (translational part \f$ \upsilon \f$
   *               and rotation vector \f$ \omega \f$)
   *
   * Computes the logarithmic, the inverse of the group exponential.
   * To be specific, this function computes \f$ \log({\cdot})^\vee \f$
   * with \f$ \vee(\cdot) \f$ being the matrix logarithm
   * and \f$ \vee{\cdot} \f$ the vee()-operator of SE2.
   *
   * \see exp()
   * \see vee()
   */
  inline static
  const Matrix<Scalar,3,1> log(const SE2Group<Scalar> & other) {
    Matrix<Scalar,3,1> upsilon_theta;
    const SO2Group<Scalar> & so2 = other.so2();
    Scalar theta = SO2Group<Scalar>::log(so2);
    upsilon_theta[2] = theta;
    Scalar halftheta = static_cast<Scalar>(0.5)*theta;
    Scalar halftheta_by_tan_of_halftheta;

    const Matrix<Scalar,2,1> & z = so2.unit_complex();
    Scalar real_minus_one = z.x()-static_cast<Scalar>(1.);
    if (abs(real_minus_one)<SophusConstants<Scalar>::epsilon()) {
      halftheta_by_tan_of_halftheta
          = static_cast<Scalar>(1.)
            - static_cast<Scalar>(1./12)*theta*theta;
    } else {
      halftheta_by_tan_of_halftheta
          = -(halftheta*z.y())/(real_minus_one);
    }
    Matrix<Scalar,2,2> V_inv;
    V_inv(0,0) = halftheta_by_tan_of_halftheta; V_inv(1,0) = -halftheta;
    V_inv(0,1) = halftheta; V_inv(1,1) = halftheta_by_tan_of_halftheta;
    upsilon_theta.template head<2>() = V_inv*other.translation();
    return upsilon_theta;
  }

  /**
   * \brief vee-operator
   *
   * \param Omega 3x3-matrix representation of Lie algebra element
   * \returns     3-vector representatin of Lie algebra element
   *
   * This is the inverse of the hat()-operator.
   *
   * \see hat()
   */
  inline static
  const Matrix<Scalar,3,1> vee(const Matrix<Scalar,3,3> & Omega) {
    Matrix<Scalar,3,1> upsilon_omega;
    upsilon_omega.template head<2>() = Omega.col(2).template head<2>();
    upsilon_omega[2] = SO2Type::vee(Omega.template topLeftCorner<2,2>());
    return upsilon_omega;
  }
};

/**
 * \brief SE2 default type - Constructors and default storage for SE2 Type
 */
template<typename _Scalar, int _Options>
class SE2Group : public SE2GroupBase<SE2Group<_Scalar,_Options> > {
public:
  typedef typename internal::traits<SE2Group<_Scalar,_Options> >
  ::Scalar Scalar;
  typedef typename internal::traits<SE2Group<_Scalar,_Options> >
  ::TranslationType TranslationType;
  typedef typename internal::traits<SE2Group<_Scalar,_Options> >
  ::SO2Type SO2Type;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * \brief Default constructor
   *
   * Initialize Complex to identity rotation and translation to zero.
   */
  inline
  SE2Group()
      : translation_( TranslationType::Zero() )
  {
  }

  /**
   * \brief Copy constructor
   */
  template<typename OtherDerived> inline
  SE2Group(const SE2GroupBase<OtherDerived> & other)
    : so2_(other.so2()), translation_(other.translation()) {
  }

  /**
   * \brief Constructor from SO2 and translation vector
   */
  template<typename OtherDerived> inline
  SE2Group(const SO2GroupBase<OtherDerived> & so2,
           const Matrix<Scalar,2,1> & translation)
    : so2_(so2), translation_(translation) {
  }

  /**
   * \brief Constructor from rotation matrix and translation vector
   *
   * \pre rotation matrix need to be orthogonal with determinant of 1
   */
  inline
  SE2Group(const Matrix2d & rotation_matrix,
           const Matrix<Scalar,2,1> & translation)
    : so2_(rotation_matrix), translation_(translation) {
  }

  /**
   * \brief Constructor from rotation angle and translation vector
   */
  inline
  SE2Group(const Scalar & theta,
           const Matrix<Scalar,2,1> & translation)
    : so2_(theta), translation_(translation) {
  }

  /**
   * \brief Constructor from complex number and translation vector
   *
   * \pre complex must not be zero
   */
  inline
  SE2Group(const std::complex<Scalar> & complex,
           const Matrix<Scalar,2,1> & translation)
    : so2_(complex), translation_(translation) {
  }

  /**
   * \brief Constructor from 3x3 matrix
   *
   * \pre 2x2 sub-matrix need to be orthogonal with determinant of 1
   */
  inline
  SE2Group(const Eigen::Matrix<Scalar,3,3>& T)
    : so2_(T.template topLeftCorner<2,2>()),
      translation_(T.template block<2,1>(0,2)) {
  }

  /**
   * \returns pointer to internal data
   *
   * This provides unsafe read/write access to internal data. SE2 is represented
   * by a pair of an SO2 element (two parameters) and a translation vector (two
   * parameters). The user needs to take care of that the complex
   * stays normalized.
   *
   * /see normalize()
   */
  EIGEN_STRONG_INLINE
  Scalar* data() {
    // so2_ and translation_ are layed out sequentially with no padding
    return so2_.data();
  }

  /**
   * \returns const pointer to internal data
   *
   * Const version of data().
   */
  EIGEN_STRONG_INLINE
  const Scalar* data() const {
      // so2_ and translation_ are layed out sequentially with no padding
      return so2_.data();
  }

  /**
   * \brief Read access to SO2
   */
  EIGEN_STRONG_INLINE
  SO2Type& so2() {
    return so2_;
  }

  /**
   * \brief Read/write access to SO2
   */
  EIGEN_STRONG_INLINE
  const SO2Type& so2() const {
    return so2_;
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
  SO2Type so2_;
  TranslationType translation_;
};


} // end namespace


namespace Eigen {
/**
 * \brief Specialisation of Eigen::Map for SE2GroupBase
 *
 * Allows us to wrap SE2 Objects around POD array
 * (e.g. external c style complex)
 */
template<typename _Scalar, int _Options>
class Map<Sophus::SE2Group<_Scalar>, _Options>
    : public Sophus::SE2GroupBase<Map<Sophus::SE2Group<_Scalar>, _Options> >
{
  typedef Sophus::SE2GroupBase<Map<Sophus::SE2Group<_Scalar>, _Options> > Base;

public:
  typedef typename internal::traits<Map>::Scalar Scalar;
  typedef typename internal::traits<Map>::TranslationType TranslationType;
  typedef typename internal::traits<Map>::SO2Type SO2Type;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  EIGEN_STRONG_INLINE
  Map(Scalar* coeffs) : so2_(coeffs), translation_(coeffs+2) {
  }

  /**
   * \brief Read/write access to SO2
   */
  EIGEN_STRONG_INLINE
  SO2Type& so2() {
    return so2_;
  }

  /**
   * \brief Read access to SO2
   */
  EIGEN_STRONG_INLINE
  const SO2Type& so2() const {
    return so2_;
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
  SO2Type so2_;
  TranslationType translation_;
};

/**
 * \brief Specialisation of Eigen::Map for const SE2GroupBase
 *
 * Allows us to wrap SE2 Objects around POD array
 * (e.g. external c style complex)
 */
template<typename _Scalar, int _Options>
class Map<const Sophus::SE2Group<_Scalar>, _Options>
    : public Sophus::SE2GroupBase<
    Map<const Sophus::SE2Group<_Scalar>, _Options> > {
  typedef Sophus::SE2GroupBase<Map<const Sophus::SE2Group<_Scalar>, _Options> >
  Base;

public:
  typedef typename internal::traits<Map>::Scalar Scalar;
  typedef typename internal::traits<Map>::TranslationType TranslationType;
  typedef typename internal::traits<Map>::SO2Type SO2Type;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  EIGEN_STRONG_INLINE
  Map(const Scalar* coeffs)
    : so2_(coeffs), translation_(coeffs+4) {
  }

  EIGEN_STRONG_INLINE
  Map(const Scalar* trans_coeffs, const Scalar* rot_coeffs)
    : translation_(trans_coeffs), so2_(rot_coeffs){
  }

  /**
   * \brief Read access to SO2
   */
  EIGEN_STRONG_INLINE
  const SO2Type& so2() const {
    return so2_;
  }

  /**
   * \brief Read access to translation vector
   */
  EIGEN_STRONG_INLINE
  const TranslationType& translation() const {
    return translation_;
  }

protected:
  const SO2Type so2_;
  const TranslationType translation_;
};

}

#endif
