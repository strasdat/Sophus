#ifndef SOPHUS_SO2_HPP
#define SOPHUS_SO2_HPP

#include <complex>

// Include only the selective set of Eigen headers that we need.
// This helps when using Sophus with unusual compilers, like nvcc.
#include <Eigen/LU>

#include "rotation_matrix.hpp"
#include "types.hpp"

namespace Sophus {
template <class Scalar_, int Options = 0>
class SO2;
using SO2d = SO2<double>;
using SO2f = SO2<float>;
}  // namespace Sophus

namespace Eigen {
namespace internal {

template <class Scalar_, int Options>
struct traits<Sophus::SO2<Scalar_, Options>> {
  using Scalar = Scalar_;
  using ComplexType = Sophus::Vector2<Scalar, Options>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::SO2<Scalar_>, Options>>
    : traits<Sophus::SO2<Scalar_, Options>> {
  using Scalar = Scalar_;
  using ComplexType = Map<Sophus::Vector2<Scalar>, Options>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::SO2<Scalar_> const, Options>>
    : traits<Sophus::SO2<Scalar_, Options> const> {
  using Scalar = Scalar_;
  using ComplexType = Map<Sophus::Vector2<Scalar> const, Options>;
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
// since ``R(alpha) * R(beta) = R(beta) * R(alpha)``,  simply because ``alpha +
// beta = beta + alpha`` with ``alpha`` and ``beta`` being rotation angles
// (about the same axis).
//
// Class invairant: The 2-norm of ``unit_complex`` must be close to 1.
// Technically speaking, it must hold that:
//
//   ``|unit_complex().squaredNorm() - 1| <= Constants<Scalar>::epsilon()``.
template <class Derived>
class SO2Base {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using ComplexT = typename Eigen::internal::traits<Derived>::ComplexType;

  // Degrees of freedom of manifold, number of dimensions in tangent space (one
  // since we only have in-plane rotations).
  static int constexpr DoF = 1;
  // Number of internal parameters used (complex numbers are a tuples).
  static int constexpr num_parameters = 2;
  // Group transformations are 2x2 matrices.
  static int constexpr N = 2;
  using Transformation = Matrix<Scalar, N, N>;
  using Point = Vector2<Scalar>;
  using Line = ParametrizedLine2<Scalar>;
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
  template <class NewScalarType>
  SOPHUS_FUNC SO2<NewScalarType> cast() const {
    return SO2<NewScalarType>(unit_complex().template cast<NewScalarType>());
  }

  // This provides unsafe read/write access to internal data. SO(2) is
  // represented by a unit complex number (two parameters). When using direct
  // write access, the user needs to take care of that the complex number stays
  // normalized.
  //
  SOPHUS_FUNC Scalar* data() { return unit_complex_nonconst().data(); }

  // Const version of data() above.
  //
  SOPHUS_FUNC Scalar const* data() const { return unit_complex().data(); }

  // Returns group inverse.
  //
  SOPHUS_FUNC SO2<Scalar> inverse() const {
    return SO2<Scalar>(unit_complex().x(), -unit_complex().y());
  }

  // Logarithmic map
  //
  // Returns tangent space representation (= rotation angle) of the instance.
  //
  SOPHUS_FUNC Scalar log() const { return SO2<Scalar>::log(*this); }

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
    Scalar const& real = unit_complex().x();
    Scalar const& imag = unit_complex().y();
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
  template <class OtherDerived>
  SOPHUS_FUNC SO2Base<Derived>& operator=(SO2Base<OtherDerived> const& other) {
    unit_complex_nonconst() = other.unit_complex();
    return *this;
  }

  // Group multiplication, which is rotation concatenation.
  //
  SOPHUS_FUNC SO2<Scalar> operator*(SO2<Scalar> const& other) const {
    SO2<Scalar> result(*this);
    result *= other;
    return result;
  }

  // Group action on 3-points.
  //
  // This function rotates a 3 dimensional point ``p`` by the SO3 element
  //  ``bar_R_foo`` (= rotation matrix): ``p_bar = bar_R_foo * p_foo``.
  //
  SOPHUS_FUNC Point operator*(Point const& p) const {
    Scalar const& real = unit_complex().x();
    Scalar const& imag = unit_complex().y();
    return Point(real * p[0] - imag * p[1], imag * p[0] + real * p[1]);
  }

  // Group action on lines.
  //
  // This function rotates a parametrized line ``l(t) = o + t * d`` by the SO3
  // element:
  //
  // Both direction ``d`` and origin ``o`` are rotated as a 3 dimensional point
  //
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), (*this) * l.direction());
  }

  // In-place group multiplication.
  //
  SOPHUS_FUNC SO2Base<Derived> operator*=(SO2<Scalar> const& other) {
    Scalar lhs_real = unit_complex().x();
    Scalar lhs_imag = unit_complex().y();
    Scalar const& rhs_real = other.unit_complex().x();
    Scalar const& rhs_imag = other.unit_complex().y();
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
  SOPHUS_FUNC void setComplex(Point const& complex) {
    unit_complex_nonconst() = complex;
    normalize();
  }

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC
  ComplexT const& unit_complex() const {
    return static_cast<Derived const*>(this)->unit_complex();
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
  SOPHUS_FUNC static SO2<Scalar> exp(Tangent const& theta) {
    return SO2<Scalar>(std::cos(theta), std::sin(theta));
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
  SOPHUS_FUNC static Transformation hat(Tangent const& theta) {
    Transformation Omega;
    // clang-format off
    Omega <<
        Scalar(0),   -theta,
            theta, Scalar(0);
    // clang-format on
    return Omega;
  }

  // Lie bracket
  //
  // It returns the Lie bracket of SO(2). Since SO(2) is a commutative group,
  // the Lie bracket is simple ``0``.
  //
  SOPHUS_FUNC static Tangent lieBracket(Tangent const&, Tangent const&) {
    return Scalar(0);
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
  SOPHUS_FUNC static Tangent log(SO2<Scalar> const& other) {
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
  // Precondition: ``Omega`` must have the following structure:
  //
  //                |  0 -a |
  //                |  a  0 |
  //
  SOPHUS_FUNC static Tangent vee(Transformation const& Omega) {
    using std::abs;
    return Omega(1, 0);
  }

 private:
  // Mutator of unit_complex is private to ensure class invariant. That is
  // the complex number must stay close to unit length.
  //
  SOPHUS_FUNC
  ComplexT& unit_complex_nonconst() {
    return static_cast<Derived*>(this)->unit_complex_nonconst();
  }
};

// SO2 default type - Constructors and default storage for SO2 Type
template <class Scalar_, int Options>
class SO2 : public SO2Base<SO2<Scalar_, Options>> {
  using Base = SO2Base<SO2<Scalar_, Options>>;

 public:
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using ComplexMember = Vector2<Scalar, Options>;

  // ``Base`` is friend so unit_complex_nonconst can be accessed from ``Base``.
  friend class SO2Base<SO2<Scalar, Options>>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor initialize unit complex number to identity rotation.
  //
  SOPHUS_FUNC SO2() : unit_complex_(Scalar(1), Scalar(0)) {}

  // Copy constructor
  //
  template <class OtherDerived>
  SOPHUS_FUNC SO2(SO2Base<OtherDerived> const& other)
      : unit_complex_(other.unit_complex()) {}

  // Constructor from rotation matrix
  //
  // Precondition: rotation matrix need to be orthogonal with determinant of 1.
  //
  SOPHUS_FUNC explicit SO2(Transformation const& R)
      : unit_complex_(Scalar(0.5) * (R(0, 0) + R(1, 1)),
                      Scalar(0.5) * (R(1, 0) - R(0, 1))) {
    SOPHUS_ENSURE(isOrthogonal(R), "R is not orthogonal:\n %", R);
    SOPHUS_ENSURE(R.determinant() > 0, "det(R) is not positive: %",
                  R.determinant());
  }

  // Returns closed SO2 given arbirary 2x2 matrix.
  //
  static SO2 fitToSO2(Transformation const& R) {
    return SO2(makeRotationMatrix(R));
  }

  // Constructor from pair of real and imaginary number.
  //
  // Precondition: The pair must not be close to zero.
  //
  SOPHUS_FUNC SO2(Scalar const& real, Scalar const& imag)
      : unit_complex_(real, imag) {
    Base::normalize();
  }

  // Constructor from 2-vector.
  //
  // Precondition: The vector must not be close to zero.
  //
  template <class D>
  SOPHUS_FUNC explicit SO2(Eigen::MatrixBase<D> const& complex)
      : unit_complex_(complex) {
    static_assert(std::is_same<typename D::Scalar, Scalar>::value,
                  "must be same Scalar type");
    Base::normalize();
  }

  // Draw uniform sample from SO(2) manifold.
  //
  template <class UniformRandomBitGenerator>
  static SO2 sampleUniform(UniformRandomBitGenerator& generator) {
    static_assert(IsUniformRandomBitGenerator<UniformRandomBitGenerator>::value,
                  "generator must meet the UniformRandomBitGenerator concept");
    std::uniform_real_distribution<Scalar> uniform(-Constants<Scalar>::pi(),
                                                   Constants<Scalar>::pi());
    return SO2(uniform(generator));
  }

  // Constructor from an rotation angle.
  //
  SOPHUS_FUNC explicit SO2(Scalar theta) {
    unit_complex_nonconst() = SO2<Scalar>::exp(theta).unit_complex();
  }

  // Accessor of unit complex number
  //
  SOPHUS_FUNC ComplexMember const& unit_complex() const {
    return unit_complex_;
  }

 protected:
  // Mutator of complex number is protected to ensure class invariant.
  //
  SOPHUS_FUNC ComplexMember& unit_complex_nonconst() { return unit_complex_; }

  ComplexMember unit_complex_;
};

}  // namespace Sophus

namespace Eigen {

// Specialization of Eigen::Map for ``SO2``.
//
// Allows us to wrap SO2 objects around POD array (e.g. external c style
// complex number / tuple).
template <class Scalar_, int Options>
class Map<Sophus::SO2<Scalar_>, Options>
    : public Sophus::SO2Base<Map<Sophus::SO2<Scalar_>, Options>> {
  using Base = Sophus::SO2Base<Map<Sophus::SO2<Scalar_>, Options>>;

 public:
  using Scalar = Scalar_;

  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  // ``Base`` is friend so unit_complex_nonconst can be accessed from ``Base``.
  friend class Sophus::SO2Base<Map<Sophus::SO2<Scalar_>, Options>>;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  Map(Scalar* coeffs) : unit_complex_(coeffs) {}

  // Accessor of unit complex number.
  //
  SOPHUS_FUNC
  Map<Sophus::Vector2<Scalar>, Options> const& unit_complex() const {
    return unit_complex_;
  }

 protected:
  // Mutator of unit_complex is protected to ensure class invariant.
  //
  SOPHUS_FUNC
  Map<Sophus::Vector2<Scalar>, Options>& unit_complex_nonconst() {
    return unit_complex_;
  }

  Map<Matrix<Scalar, 2, 1>, Options> unit_complex_;
};

// Specialization of Eigen::Map for ``SO2 const``.
//
// Allows us to wrap SO2 objects around POD array (e.g. external c style
// complex number / tuple).
template <class Scalar_, int Options>
class Map<Sophus::SO2<Scalar_> const, Options>
    : public Sophus::SO2Base<Map<Sophus::SO2<Scalar_> const, Options>> {
  using Base = Sophus::SO2Base<Map<Sophus::SO2<Scalar_> const, Options>>;

 public:
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar const* coeffs) : unit_complex_(coeffs) {}

  // Accessor of unit complex number.
  //
  SOPHUS_FUNC Map<Sophus::Vector2<Scalar> const, Options> const& unit_complex()
      const {
    return unit_complex_;
  }

 protected:
  // Mutator of unit_complex is protected to ensure class invariant.
  //
  Map<Matrix<Scalar, 2, 1> const, Options> const unit_complex_;
};
}

#endif  // SOPHUS_SO2_HPP
