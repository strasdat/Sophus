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

#include "sophus.hpp"

////////////////////////////////////////////////////////////////////////////
// Forward Declarations / typedefs
////////////////////////////////////////////////////////////////////////////

namespace Sophus {
template <typename _Scalar, int _Options = 0>
class SO2Group;
typedef SO2Group<double> SO2d; /**< double precision SO2 */
typedef SO2Group<float> SO2f;  /**< single precision SO2 */
}

////////////////////////////////////////////////////////////////////////////
// Eigen Traits (For querying derived types in CRTP hierarchy)
////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
// Eigen Traits (For querying derived types in CRTP hierarchy)
////////////////////////////////////////////////////////////////////////////

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
}
}

namespace Sophus {
/**
 * \brief SO2 base type - implements SO2 class but is storage agnostic
 *
 * [add more detailed description/tutorial]
 */
template <typename Derived>
class SO2GroupBase {
 public:
  /** \brief scalar type */
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  /** \brief complex number reference type */
  using ComplexReference =
      typename Eigen::internal::traits<Derived>::ComplexType&;
  /** \brief complex number const reference type */
  using ConstComplexReference =
      const typename Eigen::internal::traits<Derived>::ComplexType&;

  /** \brief degree of freedom of group
   *         (one for in-plane rotation) */
  static const int DoF = 1;
  /** \brief number of internal parameters used
   *         (unit complex number for rotation) */
  static const int num_parameters = 2;
  /** \brief group transformations are NxN matrices */
  static const int N = 2;
  /** \brief group transfomation type */
  using Transformation = Eigen::Matrix<Scalar, N, N>;
  /** \brief point type */
  using Point = Eigen::Matrix<Scalar, 2, 1>;
  /** \brief tangent vector type */
  using Tangent = Scalar;
  /** \brief adjoint transformation type */
  using Adjoint = Scalar;

  /**
   * \brief Adjoint transformation
   *
   * This function return the adjoint transformation \f$ Ad \f$ of the
   * group instance \f$ A \f$  such that for all \f$ x \f$
   * it holds that \f$ \widehat{Ad_A\cdot x} = A\widehat{x}A^{-1} \f$
   * with \f$\ \widehat{\cdot} \f$ being the hat()-operator.
   *
   * For SO2, it simply returns 1.
   */
  SOPHUS_FUNC Adjoint Adj() const { return 1; }

  /**
   * \returns copy of instance casted to NewScalarType
   */
  template <typename NewScalarType>
  SOPHUS_FUNC SO2Group<NewScalarType> cast() const {
    return SO2Group<NewScalarType>(
        unit_complex().template cast<NewScalarType>());
  }

  /**
   * \returns pointer to internal data
   *
   * This provides unsafe read/write access to internal data. SO2 is represented
   * by a complex number with unit length (two parameters). When using direct
   * write access, the user needs to take care of that the complex number stays
   * normalized.
   *
   * \see normalize()
   */
  SOPHUS_FUNC Scalar* data() { return unit_complex_nonconst().data(); }

  /**
   * \returns const pointer to internal data
   *
   * Const version of data().
   */
  SOPHUS_FUNC const Scalar* data() const { return unit_complex().data(); }

  /**
   * \returns group inverse of instance
   */
  SOPHUS_FUNC SO2Group<Scalar> inverse() const {
    return SO2Group<Scalar>(unit_complex().x(), -unit_complex().y());
  }

  /**
   * \brief Logarithmic map
   *
   * \returns tangent space representation (=rotation angle) of instance
   *
   * \see  log().
   */
  SOPHUS_FUNC Scalar log() const { return SO2Group<Scalar>::log(*this); }

  /**
   * \brief Normalize complex number
   *
   * It re-normalizes complex number to unit length.
   */
  SOPHUS_FUNC void normalize() {
    Scalar length = std::sqrt(unit_complex().x() * unit_complex().x() +
                              unit_complex().y() * unit_complex().y());
    SOPHUS_ENSURE(length >= Constants<Scalar>::epsilon(),
                  "Complex number should not be close to zero!");
    unit_complex_nonconst().x() /= length;
    unit_complex_nonconst().y() /= length;
  }

  /**
   * \returns 2x2 matrix representation of instance
   *
   * For SO2, the matrix representation is an orthogonal matrix R with det(R)=1,
   * thus the so-called rotation matrix.
   */
  SOPHUS_FUNC Transformation matrix() const {
    const Scalar& real = unit_complex().x();
    const Scalar& imag = unit_complex().y();
    Transformation R;
    R << real, -imag, imag, real;
    return R;
  }

  /**
   * \brief Assignment operator
   */
  template <typename OtherDerived>
  SOPHUS_FUNC SO2GroupBase<Derived>& operator=(
      const SO2GroupBase<OtherDerived>& other) {
    unit_complex_nonconst() = other.unit_complex();
    return *this;
  }

  /**
   * \brief Group multiplication
   * \see operator*=()
   */
  SOPHUS_FUNC SO2Group<Scalar> operator*(const SO2Group<Scalar>& other) const {
    SO2Group<Scalar> result(*this);
    result *= other;
    return result;
  }

  /**
   * \brief Group action on \f$ \mathbf{R}^2 \f$
   *
   * \param p point \f$p \in \mathbf{R}^2 \f$
   * \returns point \f$p' \in \mathbf{R}^2 \f$, rotated version of \f$p\f$
   *
   * This function rotates a point \f$ p \f$ in  \f$ \mathbf{R}^2 \f$ by the
   * SO2 transformation \f$R\f$ (=rotation matrix): \f$ p' = R\cdot p \f$.
   */
  SOPHUS_FUNC Point operator*(const Point& p) const {
    const Scalar& real = unit_complex().x();
    const Scalar& imag = unit_complex().y();
    return Point(real * p[0] - imag * p[1], imag * p[0] + real * p[1]);
  }

  /**
   * \brief In-place group multiplication
   *
   * \see operator*()
   */
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
    // unit complex number. Due to numerical precission issues, there might
    // be a small drift after pose concatenation. Hence, we need to renormalize
    // the complex number here.
    // Since squared-norm is close to 1, we do not need to calculate the costly
    // square-root, but can use an approximation around 1 (see
    // http://stackoverflow.com/a/12934750 for details).
    if (squared_norm != Scalar(1.0)) {
      unit_complex_nonconst() *= Scalar(2.0) / (Scalar(1.0) + squared_norm);
    }
    return *this;
  }

  /**
   * \brief Setter of internal unit complex number representation
   *
   * \param complex
   * \pre   the complex number must not be near zero
   *
   * The complex number is normalized to unit length.
   */
  SOPHUS_FUNC void setComplex(const Point& complex) {
    unit_complex_nonconst() = complex;
    normalize();
  }

  /**
   * \brief Accessor of unit complex number
   *
   * No direct write access is given to ensure the complex stays normalized.
   */
  SOPHUS_FUNC
  ConstComplexReference unit_complex() const {
    return static_cast<const Derived*>(this)->unit_complex();
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  /**
   * \brief Group exponential
   *
   * \param theta tangent space element (=rotation angle \f$ \theta \f$)
   * \returns     corresponding element of the group SO2
   *
   * To be more specific, this function computes \f$ \exp(\widehat{\theta}) \f$
   * with \f$ \exp(\cdot) \f$ being the matrix exponential
   * and \f$ \widehat{\cdot} \f$ the hat()-operator of SO2.
   *
   * \see hat()
   * \see log()
   */
  SOPHUS_FUNC static SO2Group<Scalar> exp(const Tangent& theta) {
    return SO2Group<Scalar>(std::cos(theta), std::sin(theta));
  }

  /**
   * \brief Generator
   *
   * The infinitesimal generator of SO2
   * is \f$
   *        G_0 = \left( \begin{array}{ccc}
   *                          0& -1& \\
   *                          1&  0&
   *                     \end{array} \right).
   * \f$
   * \see hat()
   */
  SOPHUS_FUNC static Transformation generator() { return hat(1); }

  /**
   * \brief hat-operator
   *
   * \param theta scalar representation of Lie algebra element
   * \returns     2x2-matrix representatin of Lie algebra element
   *
   * Formally, the hat-operator of SO2 is defined
   * as \f$ \widehat{\cdot}: \mathbf{R}^2 \rightarrow \mathbf{R}^{2\times 2},
   * \quad \widehat{\theta} = G_0\cdot \theta \f$
   * with \f$ G_0 \f$ being the infinitesial generator().
   *
   * \see generator()
   * \see vee()
   */
  SOPHUS_FUNC static Transformation hat(const Tangent& theta) {
    Transformation Omega;
    Omega << static_cast<Scalar>(0), -theta, theta, static_cast<Scalar>(0);
    return Omega;
  }

  /**
   * \brief Lie bracket
   *
   * \returns      zero
   *
   * It computes the bracket. For the Lie algebra so2, the Lie bracket is
   * simply \f$ [\theta_1, \theta_2]_{so2} = 0 \f$ since SO2 is a
   * commutative group.
   *
   * \see hat()
   * \see vee()
   */
  SOPHUS_FUNC static Tangent lieBracket(const Tangent&, const Tangent&) {
    return static_cast<Scalar>(0);
  }

  /**
   * \brief Logarithmic map
   *
   * \param other element of the group SO2
   * \returns     corresponding tangent space element
   *              (=rotation angle \f$ \theta \f$)
   *
   * Computes the logarithmic, the inverse of the group exponential.
   * To be specific, this function computes \f$ \log({\cdot})^\vee \f$
   * with \f$ \vee(\cdot) \f$ being the matrix logarithm
   * and \f$ \vee{\cdot} \f$ the vee()-operator of SO2.
   *
   * \see exp()
   * \see vee()
   */
  SOPHUS_FUNC static Tangent log(const SO2Group<Scalar>& other) {
    // todo: general implementation for Scalar not being float or double.
    return atan2(other.unit_complex_.y(), other.unit_complex().x());
  }

  /**
   * \brief vee-operator
   *
   * \param Omega 2x2-matrix representation of Lie algebra element
   * \pre         Omega need to be a skew-symmetric matrix
   * \returns     scalar representatin of Lie algebra element
   *s
   * This is the inverse of the hat()-operator.
   *
   * \see hat()
   */
  SOPHUS_FUNC static Tangent vee(const Transformation& Omega) {
    return static_cast<Scalar>(0.5) * (Omega(1, 0) - Omega(0, 1));
  }

 private:
  // Mutator of complex number is private so users are hampered
  // from setting non-unit complex numbers.
  SOPHUS_FUNC
  ComplexReference unit_complex_nonconst() {
    return static_cast<Derived*>(this)->unit_complex_nonconst();
  }
};

/**
 * \brief SO2 default type - Constructors and default storage for SO2 Type
 */
template <typename _Scalar, int _Options>
class SO2Group : public SO2GroupBase<SO2Group<_Scalar, _Options>> {
  typedef SO2GroupBase<SO2Group<_Scalar, _Options>> Base;

 public:
  /** \brief scalar type */
  using Scalar =
      typename Eigen::internal::traits<SO2Group<_Scalar, _Options>>::Scalar;
  /** \brief complex number reference type */
  using ComplexReference = typename Eigen::internal::traits<
      SO2Group<_Scalar, _Options>>::ComplexType&;
  /** \brief complex number const reference type */
  using ConstComplexReference = const typename Eigen::internal::traits<
      SO2Group<_Scalar, _Options>>::ComplexType&;

  /** \brief group transfomation type */
  using Transformation = typename Base::Transformation;
  /** \brief point type */
  using Point = typename Base::Point;
  /** \brief tangent vector type */
  using Tangent = typename Base::Tangent;
  /** \brief adjoint transformation type */
  using Adjoint = typename Base::Adjoint;

  // base is friend so unit_complex_nonconst can be accessed from base
  friend class SO2GroupBase<SO2Group<_Scalar, _Options>>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * \brief Default constructor
   *
   * Initialize complex number to identity rotation.
   */
  SOPHUS_FUNC SO2Group()
      : unit_complex_(static_cast<Scalar>(1), static_cast<Scalar>(0)) {}

  /**
   * \brief Copy constructor
   */
  template <typename OtherDerived>
  SOPHUS_FUNC SO2Group(const SO2GroupBase<OtherDerived>& other)
      : unit_complex_(other.unit_complex()) {}

  /**
   * \brief Constructor from rotation matrix
   *
   * \pre rotation matrix need to be orthogonal with determinant of 1
   */
  SOPHUS_FUNC explicit SO2Group(const Transformation& R)
      : unit_complex_(static_cast<Scalar>(0.5) * (R(0, 0) + R(1, 1)),
                      static_cast<Scalar>(0.5) * (R(1, 0) - R(0, 1))) {
    SOPHUS_ENSURE(std::abs(R.determinant() - static_cast<Scalar>(1)) <=
                      Constants<Scalar>::epsilon(),
                  "det(R) should be (close to) 1.");
  }

  /**
   * \brief Constructor from pair of real and imaginary number
   *
   * \pre pair must not be zero
   */
  SOPHUS_FUNC SO2Group(const Scalar& real, const Scalar& imag)
      : unit_complex_(real, imag) {
    Base::normalize();
  }

  /**
   * \brief Constructor from 2-vector
   *
   * \pre vector must not be zero
   */
  SOPHUS_FUNC explicit SO2Group(const Eigen::Matrix<Scalar, 2, 1>& complex)
      : unit_complex_(complex) {
    Base::normalize();
  }

  /**
   * \brief Constructor from std::complex
   *
   * \pre complex number must not be zero
   */
  SOPHUS_FUNC explicit SO2Group(const std::complex<Scalar>& complex)
      : unit_complex_(complex.real(), complex.imag()) {
    Base::normalize();
  }

  /**
   * \brief Constructor from an angle
   */
  SOPHUS_FUNC explicit SO2Group(Scalar theta) {
    unit_complex_nonconst() = SO2Group<Scalar>::exp(theta).unit_complex();
  }

  /**
   * \brief Accessor of unit complex number
   *
   * No direct write access is given to ensure the complex number stays
   * normalized.
   */
  SOPHUS_FUNC
  ConstComplexReference unit_complex() const { return unit_complex_; }

 protected:
  // Mutator of complex number is protected so users are hampered
  // from setting non-unit complex numbers.
  SOPHUS_FUNC
  ComplexReference unit_complex_nonconst() { return unit_complex_; }

  static bool isNearZero(const Scalar& real, const Scalar& imag) {
    return (real * real + imag * imag < Constants<Scalar>::epsilon());
  }

  Eigen::Matrix<Scalar, 2, 1> unit_complex_;
};

}  // end namespace

namespace Eigen {
/**
 * \brief Specialisation of Eigen::Map for SO2GroupBase
 *
 * Allows us to wrap SO2 Objects around POD array
 * (e.g. external c style complex number)
 */
template <typename _Scalar, int _Options>
class Map<Sophus::SO2Group<_Scalar>, _Options>
    : public Sophus::SO2GroupBase<Map<Sophus::SO2Group<_Scalar>, _Options>> {
  typedef Sophus::SO2GroupBase<Map<Sophus::SO2Group<_Scalar>, _Options>> Base;

 public:
  /** \brief scalar type */
  using Scalar = typename Eigen::internal::traits<Map>::Scalar;
  /** \brief complex number reference type */
  using ComplexReference = typename Eigen::internal::traits<Map>::ComplexType&;
  /** \brief complex number const reference type */
  using ConstComplexReference =
      const typename Eigen::internal::traits<Map>::ComplexType&;

  /** \brief group transfomation type */
  using Transformation = typename Base::Transformation;
  /** \brief point type */
  using Point = typename Base::Point;
  /** \brief tangent vector type */
  using Tangent = typename Base::Tangent;
  /** \brief adjoint transformation type */
  using Adjoint = typename Base::Adjoint;

  // base is friend so unit_complex_nonconst can be accessed from base
  friend class Sophus::SO2GroupBase<Map<Sophus::SO2Group<_Scalar>, _Options>>;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  Map(Scalar* coeffs) : unit_complex_(coeffs) {}

  /**
   * \brief Accessor of unit complex number
   *
   * No direct write access is given to ensure the complex number stays
   * normalized.
   */
  SOPHUS_FUNC
  ConstComplexReference unit_complex() const { return unit_complex_; }

 protected:
  // Mutator of complex number is protected so users are hampered
  // from setting non-unit complex number.
  SOPHUS_FUNC
  ComplexReference unit_complex_nonconst() { return unit_complex_; }

  Map<Matrix<Scalar, 2, 1>, _Options> unit_complex_;
};

/**
 * \brief Specialisation of Eigen::Map for const SO2GroupBase
 *
 * Allows us to wrap SO2 Objects around POD array
 * (e.g. external c style complex number)
 */
template <typename _Scalar, int _Options>
class Map<const Sophus::SO2Group<_Scalar>, _Options>
    : public Sophus::SO2GroupBase<
          Map<const Sophus::SO2Group<_Scalar>, _Options>> {
  typedef Sophus::SO2GroupBase<Map<const Sophus::SO2Group<_Scalar>, _Options>>
      Base;

 public:
  /** \brief scalar type */
  typedef typename internal::traits<Map>::Scalar Scalar;
  /** \brief complex number const reference type */
  typedef const typename internal::traits<Map>::ComplexType&
      ConstComplexReference;

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

  SOPHUS_FUNC
  Map(const Scalar* coeffs) : unit_complex_(coeffs) {}

  /**
   * \brief Accessor of unit complex number
   *
   * No direct write access is given to ensure the complex number stays
   * normalized.
   */
  SOPHUS_FUNC
  ConstComplexReference unit_complex() const { return unit_complex_; }

 protected:
  const Map<const Matrix<Scalar, 2, 1>, _Options> unit_complex_;
};
}

#endif  // SOPHUS_SO2_HPP
