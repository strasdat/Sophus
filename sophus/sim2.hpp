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

#ifndef SOPHUS_SIM2_HPP
#define SOPHUS_SIM2_HPP

#include "rxso2.hpp"

////////////////////////////////////////////////////////////////////////////
// Forward Declarations / typedefs
////////////////////////////////////////////////////////////////////////////

namespace Sophus {
template<typename _Scalar, int _Options=0> class Sim2Group;
typedef Sim2Group<double> Sim2 EIGEN_DEPRECATED;
typedef Sim2Group<double> Sim2d; /**< double precision Sim2 */
typedef Sim2Group<float> Sim2f;  /**< single precision Sim2 */
}

////////////////////////////////////////////////////////////////////////////
// Eigen Traits (For querying derived types in CRTP hierarchy)
////////////////////////////////////////////////////////////////////////////

namespace Eigen {
namespace internal {

template<typename _Scalar, int _Options>
struct traits<Sophus::Sim2Group<_Scalar,_Options> > {
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,2,1> TranslationType;
  typedef Sophus::RxSO2Group<Scalar> RxSO2Type;
};

template<typename _Scalar, int _Options>
struct traits<Map<Sophus::Sim2Group<_Scalar>, _Options> >
    : traits<Sophus::Sim2Group<_Scalar, _Options> > {
  typedef _Scalar Scalar;
  typedef Map<Matrix<Scalar,2,1>,_Options> TranslationType;
  typedef Map<Sophus::RxSO2Group<Scalar>,_Options> RxSO2Type;
};

template<typename _Scalar, int _Options>
struct traits<Map<const Sophus::Sim2Group<_Scalar>, _Options> >
    : traits<const Sophus::Sim2Group<_Scalar, _Options> > {
  typedef _Scalar Scalar;
  typedef Map<const Matrix<Scalar,2,1>,_Options> TranslationType;
  typedef Map<const Sophus::RxSO2Group<Scalar>,_Options> RxSO2Type;
};

}
}

namespace Sophus {
using namespace Eigen;
using namespace std;

/**
 * \brief Sim2 base type - implements Sim2 class but is storage agnostic
 *
 * [add more detailed description/tutorial]
 */
template<typename Derived>
class Sim2GroupBase {
public:
  /** \brief scalar type */
  typedef typename internal::traits<Derived>::Scalar Scalar;
  /** \brief translation reference type */
  typedef typename internal::traits<Derived>::TranslationType &
  TranslationReference;
  /** \brief translation const reference type */
  typedef const typename internal::traits<Derived>::TranslationType &
  ConstTranslationReference;
  /** \brief RxSO2 reference type */
  typedef typename internal::traits<Derived>::RxSO2Type &
  RxSO2Reference;
  /** \brief RxSO2 const reference type */
  typedef const typename internal::traits<Derived>::RxSO2Type &
  ConstRxSO2Reference;

  /** \brief degree of freedom of group
    *        (two for translation, one for rotation, one for scale) */
  static const int DoF = 4;
  /** \brief number of internal parameters used
   *         (complex number for rotation and scale + translation 2-vector) */
  static const int num_parameters = 4;
  /** \brief group transformations are NxN matrices */
  static const int N = 3;
  /** \brief group transfomation type */
  typedef Matrix<Scalar,N,N> Transformation;
  /** \brief point type */
  typedef Matrix<Scalar,2,1> Point;
  /** \brief tangent vector type */
  typedef Matrix<Scalar,DoF,1> Tangent;
  /** \brief adjoint transformation type */
  typedef Matrix<Scalar,DoF,DoF> Adjoint;

  /**
   * \brief Adjoint transformation
   *
   * This function return the adjoint transformation \f$ Ad \f$ of the
   * group instance \f$ A \f$  such that for all \f$ x \f$
   * it holds that \f$ \widehat{Ad_A\cdot x} = A\widehat{x}A^{-1} \f$
   * with \f$\ \widehat{\cdot} \f$ being the hat()-operator.
   */
  inline
  const Adjoint Adj() const {
    const Matrix<Scalar,2,2> & R = rxso2().rotationMatrix();
    Adjoint res;
    res.setZero();
    res.block(0,0,2,2) = scale()*R;
    res.block(0,2,2,2) << translation().y(), -translation().x(),
                         -translation().x(), -translation().y();
    res.block(2,2,2,2).setIdentity();
    return res;
  }

  /**
   * \returns copy of instance casted to NewScalarType
   */
  template<typename NewScalarType>
  inline Sim2Group<NewScalarType> cast() const {
    return
        Sim2Group<NewScalarType>(rxso2().template cast<NewScalarType>(),
                                 translation().template cast<NewScalarType>() );
  }

  /**
   * \brief In-place group multiplication
   *
   * Same as operator*=() for Sim2.
   *
   * \see operator*()
   */
  inline
  void fastMultiply(const Sim2Group<Scalar>& other) {
    translation() += (rxso2() * other.translation());
    rxso2() *= other.rxso2();
  }

  /**
   * \returns Group inverse of instance
   */
  inline
  const Sim2Group<Scalar> inverse() const {
    const RxSO2Group<Scalar> invR = rxso2().inverse();
    return Sim2Group<Scalar>(invR, invR*(translation()
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
  const Tangent log() const {
    return log(*this);
  }

  /**
   * \returns 3x3 matrix representation of instance
   */
  inline
  const Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.setIdentity();
    homogenious_matrix.block(0,0,2,2) = rxso2().matrix();
    homogenious_matrix.col(2).head(2) = translation();
    return homogenious_matrix;
  }

  /**
   * \returns 2x3 matrix representation of instance
   *
   * It returns the two first row of matrix().
   */
  inline
  const Matrix<Scalar,2,3> matrix2x3() const {
    Matrix<Scalar,2,3> matrix;
    matrix.block(0,0,2,2) = rxso2().matrix();
    matrix.col(2) = translation();
    return matrix;
  }

  /**
   * \brief Assignment operator
   */
  template<typename OtherDerived> inline
  Sim2GroupBase<Derived>& operator=
  (const Sim2GroupBase<OtherDerived> & other) {
    rxso2() = other.rxso2();
    translation() = other.translation();
    return *this;
  }

  /**
   * \brief Group multiplication
   * \see operator*=()
   */
  inline
  const Sim2Group<Scalar> operator*(const Sim2Group<Scalar>& other) const {
    Sim2Group<Scalar> result(*this);
    result *= other;
    return result;
  }

  /**
   * \brief Group action on \f$ \mathbf{R}^2 \f$
   *
   * \param p point \f$p \in \mathbf{R}^2 \f$
   * \returns point \f$p' \in \mathbf{R}^2 \f$,
   *          rotated, scaled and translated version of \f$p\f$
   *
   * This function scales, rotates and translates point \f$ p \f$
   * in \f$ \mathbf{R}^2 \f$ by the Sim(2) transformation \f$sR,t\f$
   * (=scaled rotation matrix, translation vector): \f$ p' = sR\cdot p + t \f$.
   */
  inline
  const Point operator*(const Point & p) const {
    return rxso2()*p + translation();
  }

  /**
   * \brief In-place group multiplication
   *
   * \see operator*()
   */
  inline
  void operator*=(const Sim2Group<Scalar>& other) {
    translation() += (rxso2() * other.translation());
    rxso2() *= other.rxso2();
  }

  /**
   * \brief Mutator of quaternion
   */
  inline
  typename internal::traits<Derived>::RxSO2Type::ComplexReference
  complex() {
    return rxso2().complex();
  }

  /**
   * \brief Accessor of complex number
   */
  inline
  typename internal::traits<Derived>::RxSO2Type::ConstComplexReference
  complex() const {
    return rxso2().complex();
  }

  /**
   * \returns Rotation matrix
   *
   * deprecated: use rotationMatrix() instead.
   */
  inline
  EIGEN_DEPRECATED const Transformation rotation_matrix() const {
    return rxso2().rotationMatrix();
  }

  /**
   * \returns Rotation matrix
   */
  inline
  const Matrix<Scalar,2,2> rotationMatrix() const {
    return rxso2().rotationMatrix();
  }

  /**
   * \brief Mutator of RxSO2 group
   */
  EIGEN_STRONG_INLINE
  RxSO2Reference rxso2() {
    return static_cast<Derived*>(this)->rxso2();
  }

  /**
   * \brief Accessor of RxSO2 group
   */
  EIGEN_STRONG_INLINE
  ConstRxSO2Reference rxso2() const {
    return static_cast<const Derived*>(this)->rxso2();
  }

  /**
   * \returns scale
   */
  EIGEN_STRONG_INLINE
  const Scalar scale() const {
    return rxso2().scale();
  }

  /**
   * \brief Setter of complex number using rotation matrix, leaves scale untouched
   *
   * \param R a 2x2 rotation matrix
   * \pre       the 2x2 matrix should be orthogonal and have a determinant of 1
   */
  inline
  void setRotationMatrix
  (const Matrix<Scalar,2,2> & R) {
    rxso2().setRotationMatrix(R);
  }

  /**
   * \brief Scale setter
   */
  EIGEN_STRONG_INLINE
  void setScale(const Scalar & scale) const {
    rxso2().setScale(scale);
  }

  /**
   * \brief Setter of quaternion using scaled rotation matrix
   *
   * \param sR a 2x2 scaled rotation matrix
   * \pre        the 3x3 matrix should be "scaled orthogonal"
   *             and have a positive determinant
   */
  inline
  void setScaledRotationMatrix
  (const Matrix<Scalar,2,2> & sR) {
    rxso2().setScaledRotationMatrix(sR);
  }

  /**
   * \brief Mutator of translation vector
   */
  EIGEN_STRONG_INLINE
  TranslationReference translation() {
    return static_cast<Derived*>(this)->translation();
  }

  /**
   * \brief Accessor of translation vector
   */
  EIGEN_STRONG_INLINE
  ConstTranslationReference translation() const {
    return static_cast<const Derived*>(this)->translation();
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////
  /**
   * \param   b 4-vector representation of Lie algebra element
   * \returns   derivative of Lie bracket
   *
   * This function returns \f$ \frac{\partial}{\partial a} [a, b]_{sim2} \f$
   * with \f$ [a, b]_{sim2} \f$ being the lieBracket() of the Lie algebra sim2.
   *
   * \see lieBracket()
   */
  inline static
  const Adjoint d_lieBracketab_by_d_a(const Tangent & b) {
    const Matrix<Scalar,2,1> & upsilon = b.template head<2>();
    const Scalar & theta = b[2];
    const Scalar & sigma = b[3];

    Adjoint res;
    res.setZero();
    res.template topLeftCorner<2,2>()
        = -SO2Group<Scalar>::hat(theta)-sigma*Matrix2d::Identity();
    res(0,2) = -upsilon(1);
    res(0,3) = upsilon(0);
    res(1,2) = upsilon(0);
    res(1,3) = upsilon(1);
    return res;
  }

  /**
   * \brief Group exponential
   *
   * \param a tangent space element (4-vector)
   * \returns corresponding element of the group Sim2
   *
   * The first three components of \f$ a \f$ represent the translational
   * part \f$ \upsilon \f$ in the tangent space of Sim2, while the last three
   * components of \f$ a \f$ represents the rotation vector \f$ \omega \f$.
   *
   * To be more specific, this function computes \f$ \exp(\widehat{a}) \f$
   * with \f$ \exp(\cdot) \f$ being the matrix exponential
   * and \f$ \widehat{\cdot} \f$ the hat()-operator of Sim2.
   *
   * \see hat()
   * \see log()
   */
  inline static
  const Sim2Group<Scalar> exp(const Tangent & a) {
    const Matrix<Scalar,2,1> & upsilon = a.template head<2>();
    const Scalar theta = a[2];
    const Scalar sigma = a[3];
    RxSO2Group<Scalar> rxso2 = RxSO2Group<Scalar>::exp(a.template tail<2>());

    const Matrix<Scalar,2,2> & Omega = SO2Group<Scalar>::hat(theta);
    const Matrix<Scalar,2,2> & W = calcW(theta, sigma, rxso2.scale(), Omega);

    return Sim2Group<Scalar>(rxso2, W*upsilon);

  }

  /**
   * \brief Generators
   *
   * \pre \f$ i \in \{0,1,2,3\} \f$
   * \returns \f$ i \f$th generator \f$ G_i \f$ of Sim2
   *
   * The infinitesimal generators of Sim2 are: \f[
   *        G_0 = \left( \begin{array}{ccc}
   *                          0&  0&  1\\
   *                          0&  0&  0\\
   *                          0&  0&  0\\
   *                     \end{array} \right),
   *        G_1 = \left( \begin{array}{ccc}
   *                          0&  0&  0\\
   *                          0&  0&  1\\
   *                          0&  0&  0\\
   *                     \end{array} \right),
   *        G_2 = \left( \begin{array}{ccc}
   *                          0& -1&  0\\
   *                          1&  0&  0\\
   *                          0&  0&  0\\
   *                     \end{array} \right).
   *        G_3 = \left( \begin{array}{ccc}
   *                          1&  0&  0\\
   *                          0&  1&  0\\
   *                          0&  0&  0\\
   *                     \end{array} \right).
   * \f]
   * \see hat()
   */
  inline static
  const Transformation generator(int i) {
    if (i<0 || i>3) {
      throw SophusException("i is not in range [0,3].");
    }
    Tangent e;
    e.setZero();
    e[i] = static_cast<Scalar>(1);
    return hat(e);
  }

  /**
   * \brief hat-operator
   *
   * \param omega 4-vector representation of Lie algebra element
   * \returns     3x3-matrix representatin of Lie algebra element
   *
   * Formally, the hat-operator of Sim2 is defined
   * as \f$ \widehat{\cdot}: \mathbf{R}^4 \rightarrow \mathbf{R}^{3\times 3},
   * \quad \widehat{\omega} = \sum_{i=0}^3 G_i \omega_i \f$
   * with \f$ G_i \f$ being the ith infinitesial generator().
   *
   * \see generator()
   * \see vee()
   */
  inline static
  const Transformation hat(const Tangent & v) {
    Transformation Omega;
    Omega.template topLeftCorner<2,2>()
        = RxSO2Group<Scalar>::hat(v.template tail<2>());
    Omega.col(2).template head<2>() = v.template head<2>();
    Omega.row(2).setZero();
    return Omega;
  }

  /**
   * \brief Lie bracket
   *
   * \param a 4-vector representation of Lie algebra element
   * \param b 4-vector representation of Lie algebra element
   * \returns 4-vector representation of Lie algebra element
   *
   * It computes the bracket of Sim2. To be more specific, it
   * computes \f$ [a, b]_{sim2}
   * := [\widehat{a}, \widehat{b}]^\vee \f$
   * with \f$ [A,B] = AB-BA \f$ being the matrix
   * commutator, \f$ \widehat{\cdot} \f$ the
   * hat()-operator and \f$ (\cdot)^\vee \f$ the vee()-operator of Sim2.
   *
   * \see hat()
   * \see vee()
   */
  inline static
  const Tangent lieBracket(const Tangent & a,
                           const Tangent & b) {
    const Matrix<Scalar,2,1> & upsilon1 = a.template head<2>();
    const Matrix<Scalar,2,1> & upsilon2 = b.template head<2>();
    const Scalar & theta1 = a[2];
    const Scalar & theta2 = b[2];
    const Scalar & sigma1 = a[3];
    const Scalar & sigma2 = b[3];

    Matrix<Scalar,2,1> upsilon_diff1;
    Matrix<Scalar,2,1> upsilon_diff2;

    upsilon_diff1 << sigma1*upsilon2[0]-theta1*upsilon2[1],
                     theta1*upsilon2[0]+sigma1*upsilon2[1];

    upsilon_diff2 << sigma2*upsilon1[0]-theta2*upsilon1[1],
                     theta2*upsilon1[0]+sigma2*upsilon1[1];

    Tangent res;
    res.template head<2>() = upsilon_diff1 - upsilon_diff2;
    res[2] = static_cast<Scalar>(0);
    res[3] = static_cast<Scalar>(0);

    return res;
  }

  /**
   * \brief Logarithmic map
   *
   * \param other element of the group Sim2
   * \returns     corresponding tangent space element
   *              (translational part \f$ \upsilon \f$
   *               and rotation vector \f$ \omega \f$)
   *
   * Computes the logarithmic, the inverse of the group exponential.
   * To be specific, this function computes \f$ \log({\cdot})^\vee \f$
   * with \f$ \vee(\cdot) \f$ being the matrix logarithm
   * and \f$ \vee{\cdot} \f$ the vee()-operator of Sim2.
   *
   * \see exp()
   * \see vee()
   */
  inline static
  const Tangent log(const Sim2Group<Scalar> & other) {
    Tangent res;

    const Matrix<Scalar,2,1> & theta_sigma
        = RxSO2Group<Scalar>::log(other.rxso2());

    const Scalar & theta = theta_sigma(0);
    const Scalar & sigma = theta_sigma(1);

    const Matrix<Scalar,2,2> & W
        = calcW(theta, sigma, other.scale(), SO2Group<Scalar>::hat(theta));
    res.template head<2>() = W.partialPivLu().solve(other.translation());
    res[2] = theta;
    res[3] = sigma;
    return res;
  }

  /**
   * \brief vee-operator
   *
   * \param Omega 3x3-matrix representation of Lie algebra element
   * \returns     4-vector representatin of Lie algebra element
   *
   * This is the inverse of the hat()-operator.
   *
   * \see hat()
   */
  inline static
  const Tangent vee(const Transformation & Omega) {
    Tangent upsilon_omega_sigma;
    upsilon_omega_sigma.template head<2>()
        = Omega.col(2).template head<2>();
    upsilon_omega_sigma.template tail<2>()
        = RxSO2Group<Scalar>::vee(Omega.template topLeftCorner<2,2>());
    return upsilon_omega_sigma;
  }

private:
  static
  Matrix<Scalar,2,2> calcW(const Scalar & theta,
                           const Scalar & sigma,
                           const Scalar & scale,
                           const Matrix<Scalar,2,2> & Omega){
    static const Matrix<Scalar,2,2> I
        = Matrix<Scalar,2,2>::Identity();
    static const Scalar one = static_cast<Scalar>(1.);
    static const Scalar half = static_cast<Scalar>(1./2.);
    Matrix<Scalar,2,2> Omega2 = Omega*Omega;

    Scalar A,B,C;
    if (std::abs(sigma)<SophusConstants<Scalar>::epsilon()) {
      C = one;
      if (std::abs(theta)<SophusConstants<Scalar>::epsilon()) {
        A = half;
        B = static_cast<Scalar>(1./6.);
      } else {
        Scalar theta_sq = theta*theta;
        A = (one-std::cos(theta))/theta_sq;
        B = (theta-std::sin(theta))/(theta_sq*theta);
      }
    } else {
      C = (scale-one)/sigma;
      if (std::abs(theta)<SophusConstants<Scalar>::epsilon()) {
        Scalar sigma_sq = sigma*sigma;
        A = ((sigma-one)*scale+one)/sigma_sq;
        B = ((half*sigma*sigma-sigma+one)*scale)/(sigma_sq*sigma);
      } else {
        Scalar theta_sq = theta*theta;
        Scalar a = scale*std::sin(theta);
        Scalar b = scale*std::cos(theta);
        Scalar c = theta_sq+sigma*sigma;
        A = (a*sigma+ (one-b)*theta)/(theta*c);
        B = (C-((b-one)*sigma+a*theta)/(c))*one/(theta_sq);
      }
    }
    return A*Omega + B*Omega2 + C*I;
  }
};

/**
 * \brief Sim2 default type - Constructors and default storage for Sim2 Type
 */
template<typename _Scalar, int _Options>
class Sim2Group : public Sim2GroupBase<Sim2Group<_Scalar,_Options> > {
  typedef Sim2GroupBase<Sim2Group<_Scalar,_Options> > Base;
public:
  /** \brief scalar type */
  typedef typename internal::traits<Sim2Group<_Scalar,_Options> >
  ::Scalar Scalar;
  /** \brief RxSO2 reference type */
  typedef typename internal::traits<Sim2Group<_Scalar,_Options> >
  ::RxSO2Type & RxSO2Reference;
  /** \brief RxSO2 const reference type */
  typedef const typename internal::traits<Sim2Group<_Scalar,_Options> >
  ::RxSO2Type & ConstRxSO2Reference;
  /** \brief translation reference type */
  typedef typename internal::traits<Sim2Group<_Scalar,_Options> >
  ::TranslationType & TranslationReference;
  /** \brief translation const reference type */
  typedef const typename internal::traits<Sim2Group<_Scalar,_Options> >
  ::TranslationType & ConstTranslationReference;

  /** \brief degree of freedom of group */
  static const int DoF = Base::DoF;
  /** \brief number of internal parameters used */
  static const int num_parameters = Base::num_parameters;
  /** \brief group transformations are NxN matrices */
  static const int N = Base::N;
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
   * Initialize complex number to identity rotation and translation to zero.
   */
  inline
  Sim2Group()
    : translation_( Matrix<Scalar,2,1>::Zero() )
  {
  }

  /**
   * \brief Copy constructor
   */
  template<typename OtherDerived> inline
  Sim2Group(const Sim2GroupBase<OtherDerived> & other)
    : rxso2_(other.rxso2()), translation_(other.translation()) {
  }

  /**
   * \brief Constructor from RxSO2 and translation vector
   */
  template<typename OtherDerived> inline
  Sim2Group(const RxSO2GroupBase<OtherDerived> & rxso2,
            const Point & translation)
    : rxso2_(rxso2), translation_(translation) {
  }

  /**
   * \brief Constructor from complex number and translation vector
   *
   * \pre quaternion must not be zero
   */
  inline
  Sim2Group(const Matrix<Scalar,2,1> & complex,
            const Point & translation)
    : rxso2_(complex), translation_(translation) {
  }

  /**
   * \brief Constructor from 3x3 matrix
   *
   * \pre top-left 2x2 sub-matrix need to be "scaled orthogonal"
   *      with positive determinant of
   */
  inline explicit
  Sim2Group(const Eigen::Matrix<Scalar,3,3>& T)
    : rxso2_(T.template topLeftCorner<2,2>()),
      translation_(T.template block<2,1>(0,2)) {
  }

  /**
   * \returns pointer to internal data
   *
   * This provides unsafe read/write access to internal data. Sim2 is
   * represented by a pair of an RxSO2 element (2 parameters) and translation
   * vector (2 parameters).
   *
   */
  EIGEN_STRONG_INLINE
  Scalar* data() {
    // rxso2_ and translation_ are layed out sequentially with no padding
    return rxso2_.data();
  }

  /**
   * \returns const pointer to internal data
   *
   * Const version of data().
   */
  EIGEN_STRONG_INLINE
  const Scalar* data() const {
    // rxso2_ and translation_ are layed out sequentially with no padding
    return rxso2_.data();
  }

  /**
   * \brief Accessor of RxSO2
   */
  EIGEN_STRONG_INLINE
  RxSO2Reference rxso2() {
    return rxso2_;
  }

  /**
   * \brief Mutator of RxSO2
   */
  EIGEN_STRONG_INLINE
  ConstRxSO2Reference rxso2() const {
    return rxso2_;
  }

  /**
   * \brief Mutator of translation vector
   */
  EIGEN_STRONG_INLINE
  TranslationReference translation() {
    return translation_;
  }

  /**
   * \brief Accessor of translation vector
   */
  EIGEN_STRONG_INLINE
  ConstTranslationReference translation() const {
    return translation_;
  }

protected:
  Sophus::RxSO2Group<Scalar> rxso2_;
  Matrix<Scalar,2,1> translation_;
};


} // end namespace


namespace Eigen {
/**
 * \brief Specialisation of Eigen::Map for Sim2GroupBase
 *
 * Allows us to wrap Sim2 Objects around POD array
 * (e.g. external c style quaternion)
 */
template<typename _Scalar, int _Options>
class Map<Sophus::Sim2Group<_Scalar>, _Options>
    : public Sophus::Sim2GroupBase<Map<Sophus::Sim2Group<_Scalar>, _Options> > {
  typedef Sophus::Sim2GroupBase<Map<Sophus::Sim2Group<_Scalar>, _Options> >
  Base;

public:
  /** \brief scalar type */
  typedef typename internal::traits<Map>::Scalar Scalar;
  /** \brief translation reference type */
  typedef typename internal::traits<Map>::TranslationType &
  TranslationReference;
  /** \brief translation const reference type */
  typedef const typename internal::traits<Map>::TranslationType &
  ConstTranslationReference;
  /** \brief RxSO2 reference type */
  typedef typename internal::traits<Map>::RxSO2Type &
  RxSO2Reference;
  /** \brief RxSO2 const reference type */
  typedef const typename internal::traits<Map>::RxSO2Type &
  ConstRxSO2Reference;


  /** \brief degree of freedom of group */
  static const int DoF = Base::DoF;
  /** \brief number of internal parameters used */
  static const int num_parameters = Base::num_parameters;
  /** \brief group transformations are NxN matrices */
  static const int N = Base::N;
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

  EIGEN_STRONG_INLINE
  Map(Scalar* coeffs)
    : rxso2_(coeffs),
      translation_(coeffs+Sophus::RxSO2Group<Scalar>::num_parameters) {
  }

  /**
   * \brief Mutator of RxSO2
   */
  EIGEN_STRONG_INLINE
  RxSO2Reference rxso2() {
    return rxso2_;
  }

  /**
   * \brief Accessor of RxSO2
   */
  EIGEN_STRONG_INLINE
  ConstRxSO2Reference rxso2() const {
    return rxso2_;
  }

  /**
   * \brief Mutator of translation vector
   */
  EIGEN_STRONG_INLINE
  TranslationReference translation() {
    return translation_;
  }

  /**
   * \brief Accessor of translation vector
   */
  EIGEN_STRONG_INLINE
  ConstTranslationReference translation() const {
    return translation_;
  }

protected:
  Map<Sophus::RxSO2Group<Scalar>,_Options> rxso2_;
  Map<Matrix<Scalar,2,1>,_Options> translation_;
};

/**
 * \brief Specialisation of Eigen::Map for const Sim2GroupBase
 *
 * Allows us to wrap Sim2 Objects around POD array
 * (e.g. external c style quaternion)
 */
template<typename _Scalar, int _Options>
class Map<const Sophus::Sim2Group<_Scalar>, _Options>
    : public Sophus::Sim2GroupBase<
    Map<const Sophus::Sim2Group<_Scalar>, _Options> > {
  typedef Sophus::Sim2GroupBase<
  Map<const Sophus::Sim2Group<_Scalar>, _Options> > Base;

public:
  /** \brief scalar type */
  typedef typename internal::traits<Map>::Scalar Scalar;
  /** \brief translation type */
  typedef const typename internal::traits<Map>::TranslationType &
  ConstTranslationReference;
  /** \brief RxSO2 const reference type */
  typedef const typename internal::traits<Map>::RxSO2Type &
  ConstRxSO2Reference;

  /** \brief degree of freedom of group */
  static const int DoF = Base::DoF;
  /** \brief number of internal parameters used */
  static const int num_parameters = Base::num_parameters;
  /** \brief group transformations are NxN matrices */
  static const int N = Base::N;
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

  EIGEN_STRONG_INLINE
  Map(const Scalar* coeffs)
    : rxso2_(coeffs),
      translation_(coeffs+Sophus::RxSO2Group<Scalar>::num_parameters) {
  }

  EIGEN_STRONG_INLINE
  Map(const Scalar* trans_coeffs, const Scalar* rot_coeffs)
    : translation_(trans_coeffs), rxso2_(rot_coeffs){
  }

  /**
   * \brief Accessor of RxSO2
   */
  EIGEN_STRONG_INLINE
  ConstRxSO2Reference rxso2() const {
    return rxso2_;
  }

  /**
   * \brief Accessor of translation vector
   */
  EIGEN_STRONG_INLINE
  ConstTranslationReference translation() const {
    return translation_;
  }

protected:
  const Map<const Sophus::RxSO2Group<Scalar>,_Options> rxso2_;
  const Map<const Matrix<Scalar,2,1>,_Options> translation_;
};

}

#endif
