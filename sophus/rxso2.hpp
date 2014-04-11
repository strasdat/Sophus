// This file is part of Sophus.
//
// Copyright 2013-2014 Ping-Lin Chang
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

#ifndef SOPHUS_RXSO2_HPP
#define SOPHUS_RXSO2_HPP

#include "sophus.hpp"
#include "so2.hpp"

////////////////////////////////////////////////////////////////////////////
// Forward Declarations / typedefs
////////////////////////////////////////////////////////////////////////////

namespace Sophus {
template<typename _Scalar, int _Options=0> class RxSO2Group;
typedef RxSO2Group<double> RxSO2 EIGEN_DEPRECATED;
typedef RxSO2Group<double> RxSO2d; /**< double precision RxSO2 */
typedef RxSO2Group<float> RxSO2f;  /**< single precision RxSO2 */
}

////////////////////////////////////////////////////////////////////////////
// Eigen Traits (For querying derived types in CRTP hierarchy)
////////////////////////////////////////////////////////////////////////////

namespace Eigen {
namespace internal {

template<typename _Scalar, int _Options>
struct traits<Sophus::RxSO2Group<_Scalar,_Options> > {
    typedef _Scalar Scalar;
    typedef Matrix<Scalar,2,1> ComplexType;
};

template<typename _Scalar, int _Options>
struct traits<Map<Sophus::RxSO2Group<_Scalar>, _Options> >
        : traits<Sophus::RxSO2Group<_Scalar, _Options> > {
    typedef _Scalar Scalar;
    typedef Map<Matrix<Scalar,2,1>,_Options> ComplexType;
};

template<typename _Scalar, int _Options>
struct traits<Map<const Sophus::RxSO2Group<_Scalar>, _Options> >
        : traits<const Sophus::RxSO2Group<_Scalar, _Options> > {
    typedef _Scalar Scalar;
    typedef Map<const Matrix<Scalar,2,1>,_Options> ComplexType;
};

}
}

namespace Sophus {
using namespace Eigen;

/**
 * \brief RxSO2 base type - implements RxSO2 class but is storage agnostic
 *
 * This class implements the group \f$ R^{+} \times SO(2) \f$ (RxSO2), the direct
 * product of the group of positive scalar matrices (=isomorphy to the positive
 * real numbers) and the two-dimensional special orthogonal group SO(2).
 * Geometrically, it is the group of rotation and scaling in two dimensions.
 * As a matrix groups, RxSO2 consists of matrices of the form \f$ s\cdot R \f$
 * where \f$ R \f$ is an orthognal matrix with \f$ det(R)=1 \f$ and \f$ s>0 \f$
 * be a positive real number.
 *
 * Internally, RxSO2 is represented by the group of non-zero complex number. This
 * is a most compact representation since the degrees of freedom (DoF) of RxSO2
 * (=2) equals the number of internal parameters (=2).
 *
 * [add more detailed description/tutorial]
 */
template<typename Derived>
class RxSO2GroupBase {
public:
    /** \brief scalar type */
    typedef typename internal::traits<Derived>::Scalar Scalar;
    /** \brief complex number reference type */
    typedef typename internal::traits<Derived>::ComplexType &
    ComplexReference;
    /** \brief complex number const reference type */
    typedef const typename internal::traits<Derived>::ComplexType &
    ConstComplexReference;


    /** \brief degree of freedom of group
   *         (one for rotation and one for scaling) */
    static const int DoF = 2;
    /** \brief number of internal parameters used
   *         (complex number for rotation and scaling) */
    static const int num_parameters = 2;
    /** \brief group transformations are NxN matrices */
    static const int N = 2;
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
   *
   * For RxSO2, it simply returns the rotation matrix corresponding to
   * \f$ A \f$.
   */
    inline
    const Adjoint Adj() const {
        Adjoint res;
        res.setIdentity();
        return res;
    }

    /**
   * \returns copy of instance casted to NewScalarType
   */
    template<typename NewScalarType>
    inline RxSO2Group<NewScalarType> cast() const {
        return RxSO2Group<NewScalarType>(complex()
                                         .template cast<NewScalarType>() );
    }

    /*!
     * \returns pointer to internal data
     *
     * This provides unsafe read/write access to internal data. RxSO2 is represented
     * by a scaled complex number (two parameters).
     *
     */
    inline Scalar* data() {
        return complex().data();
    }

    /*!
     * \returns const pointer to internal data
     *
     * Const version of data().
     */
    inline const Scalar* data() const {
        return complex().data();
    }

    /*!
     * \brief Fast group multiplication
     *
     * \see operator*=()
     */
    inline
    void fastMultiply(const RxSO2Group<Scalar>& other) {
        Scalar lhs_real = complex().x();
        Scalar lhs_imag = complex().y();
        const Scalar & rhs_real = other.complex().x();
        const Scalar & rhs_imag = other.complex().y();

        // complex multiplication
        complex().x() = lhs_real*rhs_real - lhs_imag*rhs_imag;
        complex().y() = lhs_real*rhs_imag + lhs_imag*rhs_real;
    }

    /**
   * \returns group inverse of instance
   */
    inline
    const RxSO2Group<Scalar> inverse() const {
        const Scalar scale = complex().norm();
        const Scalar sq_scale = scale*scale;
        return RxSO2Group<Scalar>(complex().x()/sq_scale, -complex().y()/sq_scale);
    }

    /*!
     * \brief Logarithmic map
     *
     * \returns tangent space representation (=rotation vector) of instance
     *
     * \see  log().
     */
    inline
    const Tangent log() const {
        return RxSO2Group<Scalar>::log(*this);
    }

    /*!
     * \returns 2x2 matrix representation of instance
     *
     * For RxSO2, the matrix representation is a scaled orthogonal
     * matrix \f$ sR \f$ with \f$ det(sR)=s^2 \f$, thus a scaled rotation
     * matrix \f$ R \f$  with scale s.
     */
    inline
    const Transformation matrix() const {
        const Scalar & real = complex().x();
        const Scalar & imag = complex().y();
        Transformation R;
        R << real, -imag
            ,imag,  real;
        return R;
    }

    /*!
     * \brief Assignment operator
     */
    template<typename OtherDerived> inline
    RxSO2GroupBase<Derived>& operator=
    (const RxSO2GroupBase<OtherDerived> & other) {
        complex() = other.complex();
        return *this;
    }

    /*!
     * \brief Group multiplication
     * \see operator*=()
     */
    inline
    const RxSO2Group<Scalar> operator*(const RxSO2Group<Scalar>& other) const {
        RxSO2Group<Scalar> result(*this);
        result *= other;
        return result;
    }

    /*!
     * \brief Group action on \f$ \mathbf{R}^2 \f$
     *
     * \param p point \f$p \in \mathbf{R}^2 \f$
     * \returns point \f$p' \in \mathbf{R}^2 \f$,
     *          rotated and scaled version of \f$p\f$
     *
     * This function rotates and scales a point \f$ p \f$ in  \f$ \mathbf{R}^2 \f$
     * by the RxSO2 transformation \f$sR\f$ (=rotation matrix)
     * : \f$ p' = sR\cdot p \f$.
     */
    inline
    const Point operator*(const Point & p) const {
        return matrix()*p;
    }

    /*!
     * \brief In-place group multiplication
     * \see operator*=()
     */
    inline
    void operator*=(const RxSO2Group<Scalar>& other) {
        fastMultiply(other);
    }

    /*!
     * \brief Mutator of complex number
     */
    EIGEN_STRONG_INLINE
    ComplexReference complex() {
        return static_cast<Derived*>(this)->complex();
    }

    /*!
     * \brief Accessor of complex number
     */
    EIGEN_STRONG_INLINE
    ConstComplexReference complex() const {
        return static_cast<const Derived*>(this)->complex();
    }

    /**
     * \returns rotation matrix
     */
    inline
    Transformation rotationMatrix() const {
      const Scalar scale = complex().norm();
      const Scalar real = complex().x()/scale;
      const Scalar imag = complex().y()/scale;

      Transformation R;
      R << real, -imag,
           imag, real;

      return R;
    }

    /*!
     * \returns scale
     */
    EIGEN_STRONG_INLINE
    const Scalar scale() const {
        return complex().norm();
    }

    /**
     * \brief Setter of complex number using rotation matrix, leaves scale untouched
     *
     * \param R a 2x2 rotation matrix
     * \pre       the 2x2 matrix should be orthogonal and have a determinant of 1
     */
    inline
    void setRotationMatrix(const Transformation & R) {

        SOPHUS_ENSURE(std::abs(R.determinant()-static_cast<Scalar>(1))
                      < SophusConstants<Scalar>::epsilon(),
                      "det(R) is not near 1");

        Scalar saved_scale = scale();
        complex().x() = R(0,0);
        complex().y() = R(1,1);
        complex() *= saved_scale;
    }

    /**
     * \brief Scale setter
     */
    EIGEN_STRONG_INLINE
    void setScale(const Scalar & new_scale) {
        Scalar scale = complex().norm();

        SOPHUS_ENSURE(new_scale
                      > SophusConstants<Scalar>::epsilon(),
                      "Scale factor should be positive");

        complex().x() /= scale;
        complex().y() /= scale;

        complex() *= new_scale;
    }

    /**
     * \brief Setter of complex number using scaled rotation matrix
     *
     * \param sR a 2x2 scaled rotation matrix
     * \pre        the 2x2 matrix should be "scaled orthogonal"
     *             and have a positive determinant
     */
    inline
    void setScaledRotationMatrix(const Transformation & sR) {

      const Scalar scale = sqrt(sR(0,0)*sR(0,0)+sR(1,1)*sR(1,1));
      Transformation R = sR/scale;

      SOPHUS_ENSURE(std::abs(R.determinant()-static_cast<Scalar>(1))
                    < SophusConstants<Scalar>::epsilon(),
                    "det(R) is not near 1");

      complex().x() = sR(0,0);
      complex().y() = sR(1,1);

    }

    ////////////////////////////////////////////////////////////////////////////
    // public static functions
    ////////////////////////////////////////////////////////////////////////////
    /*!
     * \brief Group exponential
     *
     * \param a tangent space element
     *          (rotation vector \f$ \omega \f$ and logarithm of scale)
     * \returns corresponding element of the group RxSO2
     *
     * To be more specific, this function computes \f$ \exp(\widehat{a}) \f$
     * with \f$ \exp(\cdot) \f$ being the matrix exponential
     * and \f$ \widehat{\cdot} \f$ the hat()-operator of RxSO2.
     *
     * \see hat()
     * \see log()
     */
    inline static
    const RxSO2Group<Scalar> exp(const Tangent & a) {
        return RxSO2Group<Scalar>(std::exp(a[1]), SO2Group<Scalar>(a[0]));
    }

    /*!
     * \brief Generator
     *
     * \param \f$ i \in \{0,1\} \f$
     * \returns \f$ i \f$th generator \f$ G_i \f$ of RxSO2
     *
     * The infinitesimal generator of SO2
     * is \f$
     *        G_0 = \left( \begin{array}{cc}
     *                          0& -1& \\
     *                          1&  0&
     *                     \end{array} \right),
     *        G_1 = \left( \begin{array}{cc}
     *                          1&  0& \\
     *                          0&  1&
     *                     \end{array} \right).
     * \f$
     * \see hat()
     */
    inline static
    const Transformation generator(int i) {
        Tangent e;
        e.setZero();
        e[i] = static_cast<Scalar>(1);
        return hat(e);
    }

    /**
   * \brief hat-operator
   *
   * \param a 2-vector representation of Lie algebra element
   * \returns 2x2-matrix representatin of Lie algebra element
   *
   * Formally, the hat-operator of RxSO2 is defined
   * as \f$ \widehat{\cdot}: \mathbf{R}^2 \rightarrow \mathbf{R}^{2\times 2},
   * \quad \widehat{\theta} =\sum_{i=0}^1 G_i a_i  \f$
   * with \f$ G_i \f$ being the ith infinitesial generator().
   *
   * \see generator()
   * \see vee()
   */
    inline static
    const Transformation hat(const Tangent & a) {
        Transformation A;
        A <<  a(1), -a(0),
              a(0), a(1);
        return A;
    }

    /**
   * \brief Lie bracket
   *
   * \param a 2-vector representation of Lie algebra element
   * \param b 2-vector representation of Lie algebra element
   * \returns 2-vector representation of Lie algebra element
   *
   * It computes the bracket. For the Lie algebra RxSo2, the Lie bracket is
   * simply \f$ [\theta_1, \theta_2]_{rxso2} = [0, 0] \f$ since rxsO2 is a
   * commutative group.
   *
   * \see hat()
   * \see vee()
   */
    inline static
    const Tangent lieBracket(const Tangent & a,
                             const Tangent & b) {
        return Tangent(0, 0);
    }


    /*!
   * \brief Logarithmic map
   *
   * \param other element of the group RxSO2
   * \returns     corresponding tangent space element
   *              (rotation vector \f$ \omega \f$ and logarithm of scale)
   *
   * Computes the logarithmic, the inverse of the group exponential.
   * To be specific, this function computes \f$ \log({\cdot})^\vee \f$
   * with \f$ log(\cdot) \f$ being the matrix logarithm
   * and \f$ \vee \f$ the vee()-operator of RxSO2.
   *
   * \see exp()
   * \see vee()
   */
    inline static
    const Tangent log(const RxSO2Group<Scalar> & other) {
        const Scalar & scale = other.complex().norm();
        Tangent omega_sigma;
        omega_sigma[0] = SO2Group<Scalar>::log(SO2Group<Scalar>(other.complex()));
        omega_sigma[1] = std::log(scale);
        return omega_sigma;
    }

    /**
   * \brief vee-operator
   *
   * \param Omega 2x2-matrix representation of Lie algebra element
   * \returns     2-vector representatin of Lie algebra element
   *
   * This is the inverse of the hat()-operator.
   *
   * \see hat()
   */
    inline static
    const Tangent vee(const Transformation & Omega) {
        return Tangent( static_cast<Scalar>(0.5) * (Omega(1,0) - Omega(0,1)),
                        static_cast<Scalar>(0.5) * (Omega(0,0) + Omega(1,1)));
    }

};

/**
 * \brief RxSO2 default type - Constructors and default storage for RxSO2 Type
 */
template<typename _Scalar, int _Options>
class RxSO2Group : public RxSO2GroupBase<RxSO2Group<_Scalar,_Options> > {
    typedef RxSO2GroupBase<RxSO2Group<_Scalar,_Options> > Base;
public:
    /** \brief scalar type */
    typedef typename internal::traits<SO2Group<_Scalar,_Options> >
    ::Scalar Scalar;
    /** \brief quaternion reference type */
    typedef typename internal::traits<SO2Group<_Scalar,_Options> >
    ::ComplexType & ComplexReference;
    /** \brief quaternion const reference type */
    typedef const typename internal::traits<SO2Group<_Scalar,_Options> >
    ::ComplexType & ConstComplexReference;

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
     * Initialize complex number to identity rotation and scale.
     */
    inline RxSO2Group()
        : complex_(static_cast<Scalar>(1), static_cast<Scalar>(0)) {
    }

    /**
     * \brief Copy constructor
     */
    template<typename OtherDerived> inline
    RxSO2Group(const RxSO2GroupBase<OtherDerived> & other)
        : complex_(other.complex()) {
    }

    /*!
     * \brief Constructor from pair of real and imaginary number
     *
     * \pre pair must not be zero
     */
    inline RxSO2Group(const Scalar & real, const Scalar & imag)
        : complex_(real, imag) {

        SOPHUS_ENSURE(complex_.norm()
                      > SophusConstants<Scalar>::epsilon(),
                      "Scale factor should be positive");
    }

    /*!
     * \brief Constructor from 2-vector
     *
     * \pre vector must not be zero
     */
    inline explicit
    RxSO2Group(const Matrix<Scalar,2,1> & complex)
      : complex_(complex) {
    }

    /*!
     * \brief Constructor from std::complex
     *
     * \pre complex number must not be zero
     */
    inline explicit
    RxSO2Group(const std::complex<Scalar> & complex)
        : complex_(complex.real(), complex.imag()) {
        SOPHUS_ENSURE(complex_.norm()
                      > SophusConstants<Scalar>::epsilon(),
                      "Scale factor should be positive");
    }

    /*!
     * \brief Constructor from scaled rotation matrix
     *
     * \pre matrix need to be "scaled orthogonal" with positive determinant
     */
    inline explicit
    RxSO2Group(const Transformation & sR) {
        setScaledRotationMatrix(sR);
    }

    /**
     * \brief Constructor from scale factor and rotation matrix
     *
     * \pre rotation matrix need to be orthogonal with determinant of 1
     * \pre scale need to be not zero
     */
    inline explicit
    RxSO2Group(const Scalar & scale, const Transformation & R) {
//        SOPHUS_ENSURE(scale > static_cast<Scalar>(0),
//                      "Scale factor should be positive");
        RxSO2Group(scale*R);
    }

    /**
     * \brief Constructor from scale factor and SO(2)
     *
     * \pre scale need to be not zero
     */
    inline
    RxSO2Group(const Scalar & scale, const SO2Group<Scalar> & so2) {
//        SOPHUS_ENSURE(scale > static_cast<Scalar>(0),
//                      "Scale factor should be positive");

        complex_ = scale*so2.unit_complex();
    }

    /**
     * \brief Mutator of complex number
     */
    EIGEN_STRONG_INLINE
    ComplexReference complex() {
        return complex_;
    }

    /**
     * \brief Accessor of complex number
     */
    EIGEN_STRONG_INLINE
    ConstComplexReference complex() const {
        return complex_;
    }

    static bool isNearZero(const Scalar & real, const Scalar & imag) {
        return (real*real + imag*imag < SophusConstants<Scalar>::epsilon());
    }

    Matrix<Scalar,2,1> complex_;
};

} // end namespace

namespace Eigen {
/**
 * \brief Specialisation of Eigen::Map for RxSO2GroupBase
 *
 * Allows us to wrap RxSO2 Objects around POD array
 * (e.g. external c style complex number)
 */
template<typename _Scalar, int _Options>
class Map<Sophus::RxSO2Group<_Scalar>, _Options>
        : public Sophus::RxSO2GroupBase<
        Map<Sophus::RxSO2Group<_Scalar>,_Options> > {
    typedef Sophus::RxSO2GroupBase<Map<Sophus::RxSO2Group<_Scalar>, _Options> >
    Base;

public:
    /** \brief scalar type */
    typedef typename internal::traits<Map>::Scalar Scalar;
    /** \brief complex number reference type */
    typedef typename internal::traits<Map>::ComplexType & ComplexReference;
    /** \brief complex number const reference type */
    typedef const typename internal::traits<Map>::ComplexType &
    ConstComplexReference;

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
    Map(Scalar* coeffs) : complex_(coeffs) {
    }

    /**
     * \brief Mutator of complex number
     */
    EIGEN_STRONG_INLINE
    ComplexReference complex() {
        return complex_;
    }

    /**
     * \brief Accessor of quaternion
     */
    EIGEN_STRONG_INLINE
    ConstComplexReference complex() const {
        return complex_;
    }

protected:
    Map<Matrix<Scalar,2,1>,_Options> complex_;
};

/**
 * \brief Specialisation of Eigen::Map for const RxSO2GroupBase
 *
 * Allows us to wrap RxSO2 Objects around POD array
 * (e.g. external c style complex number)
 */
template<typename _Scalar, int _Options>
class Map<const Sophus::RxSO2Group<_Scalar>, _Options>
        : public Sophus::RxSO2GroupBase<
        Map<const Sophus::RxSO2Group<_Scalar>, _Options> > {
    typedef Sophus::RxSO2GroupBase<
    Map<const Sophus::RxSO2Group<_Scalar>, _Options> > Base;

public:
    /** \brief scalar type */
    typedef typename internal::traits<Map>::Scalar Scalar;
    /** \brief complex number const reference type */
    typedef const typename internal::traits<Map>::ComplexType &
    ConstComplexReference;


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
    Map(const Scalar* coeffs) : complex_(coeffs) {
    }

    /**
     * \brief Accessor of complex number
     *
     * No direct write access is given to ensure the complex number stays
     * normalized.
     */
    EIGEN_STRONG_INLINE
    ConstComplexReference complex() const {
        return complex_;
    }


protected:
    const Map<const Matrix<Scalar,2,1>,_Options> complex_;
};

}

#endif // SOPHUS_RXSO2_HPP
