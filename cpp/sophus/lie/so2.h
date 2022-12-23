// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Special orthogonal group SO(2) - rotation in 2d.

#pragma once

#include <type_traits>

// Include only the selective set of Eigen headers that we need.
// This helps when using Sophus with unusual compilers, like nvcc.
#include "sophus/common/types.h"
#include "sophus/linalg/rotation_matrix.h"

#include <Eigen/LU>

namespace sophus {
template <class TScalar>
class So2;
using So2F64 = So2<double>;
using So2F32 = So2<float>;

template <class TScalar>
/* [[deprecated]] */ using SO2 = So2<TScalar>;
/* [[deprecated]] */ using SO2d = So2F64;
/* [[deprecated]] */ using SO2f = So2F32;
}  // namespace sophus

namespace Eigen {  // NOLINT
namespace internal {

template <class TScalar>
struct traits<sophus::So2<TScalar>> {
  using Scalar = TScalar;
  using ComplexType = Eigen::Matrix<Scalar, 2, 1>;
};

template <class TScalar>
struct traits<Map<sophus::So2<TScalar>>> : traits<sophus::So2<TScalar>> {
  using Scalar = TScalar;
  using ComplexType = Map<Eigen::Vector2<Scalar>>;
};

template <class TScalar>
struct traits<Map<sophus::So2<TScalar> const>>
    : traits<sophus::So2<TScalar> const> {
  using Scalar = TScalar;
  using ComplexType = Map<Eigen::Vector2<Scalar> const>;
};
}  // namespace internal
}  // namespace Eigen

namespace sophus {

/// So2 base type - implements So2 class but is storage agnostic.
///
/// SO(2) is the group of rotations in 2d. As a matrix group, it is the set of
/// matrices which are orthogonal such that ``R * R' = I`` (with ``R'`` being
/// the transpose of ``R``) and have a positive determinant. In particular, the
/// determinant is 1. Let ``theta`` be the rotation angle, the rotation matrix
/// can be written in close form:
///
///      | cos(theta) -sin(theta) |
///      | sin(theta)  cos(theta) |
///
/// As a matter of fact, the first column of those matrices is isomorph to the
/// set of unit complex numbers U(1). Thus, internally, So2 is represented as
/// complex number with length 1.
///
/// SO(2) is a compact and commutative group. First it is compact since the set
/// of rotation matrices is a closed and bounded set. Second it is commutative
/// since ``R(alpha) * R(beta) = R(beta) * R(alpha)``,  simply because ``alpha +
/// beta = beta + alpha`` with ``alpha`` and ``beta`` being rotation angles
/// (about the same axis).
///
/// Class invariant: The 2-norm of ``unit_complex`` must be close to 1.
/// Technically speaking, it must hold that:
///
///   ``|unit_complex().squaredNorm() - 1| <= Constants::epsilon()``.
template <class TDerived>
class So2Base {
 public:
  using Scalar = typename Eigen::internal::traits<TDerived>::Scalar;
  using ComplexT = typename Eigen::internal::traits<TDerived>::ComplexType;
  using ComplexTemporaryType = Eigen::Matrix<Scalar, 2, 1>;

  /// Degrees of freedom of manifold, number of dimensions in tangent space (one
  /// since we only have in-plane rotations).
  static int constexpr kDoF = 1;
  /// Number of internal parameters used (complex numbers are a tuples).
  static int constexpr kNumParameters = 2;
  /// Group transformations are 2x2 matrices.
  static int constexpr kMatrixDim = 2;
  /// Points are 3-dimensional
  static int constexpr kPointDim = 2;
  using Transformation = Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim>;
  using Point = Eigen::Vector2<Scalar>;
  using HomogeneousPoint = Eigen::Vector3<Scalar>;
  using Line = Eigen::ParametrizedLine<Scalar, 2>;
  using Hyperplane = Eigen::Hyperplane<Scalar, 2>;
  using Tangent = Scalar;
  using Adjoint = Scalar;

  /// For binary operations the return type is determined with the
  /// ScalarBinaryOpTraits feature of Eigen. This allows mixing concrete and Map
  /// types, as well as other compatible scalar types such as Ceres::Jet and
  /// double scalars with So2 operations.
  template <class TOtherDerived>
  using ReturnScalar = typename Eigen::
      ScalarBinaryOpTraits<Scalar, typename TOtherDerived::Scalar>::ReturnType;

  template <class TOtherDerived>
  using So2Product = So2<ReturnScalar<TOtherDerived>>;

  template <class TPointDerived>
  using PointProduct = Eigen::Vector2<ReturnScalar<TPointDerived>>;

  template <class THPointDerived>
  using HomogeneousPointProduct = Eigen::Vector3<ReturnScalar<THPointDerived>>;

  /// Adjoint transformation
  ///
  /// This function return the adjoint transformation ``Ad`` of the group
  /// element ``A`` such that for all ``x`` it holds that
  /// ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  ///
  /// It simply ``1``, since ``SO(2)`` is a commutative group.
  ///
  SOPHUS_FUNC [[nodiscard]] Adjoint adj() const { return Scalar(1); }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class TNewScalarType>
  SOPHUS_FUNC [[nodiscard]] So2<TNewScalarType> cast() const {
    return So2<TNewScalarType>(unitComplex().template cast<TNewScalarType>());
  }

  /// This provides unsafe read/write access to internal data. SO(2) is
  /// represented by a unit complex number (two parameters). When using direct
  /// write access, the user needs to take care of that the complex number stays
  /// normalized.
  ///
  SOPHUS_FUNC Scalar* data() { return mutUnitComplex().data(); }

  /// Const version of data() above.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar const* data() const {
    return unitComplex().data();
  }

  /// Returns group inverse.
  ///
  SOPHUS_FUNC [[nodiscard]] So2<Scalar> inverse() const {
    return So2<Scalar>(unitComplex().x(), -unitComplex().y());
  }

  /// Logarithmic map
  ///
  /// Computes the logarithm, the inverse of the group exponential which maps
  /// element of the group (rotation matrices) to elements of the tangent space
  /// (rotation angles).
  ///
  /// To be specific, this function computes ``vee(logmat(.))`` with
  /// ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  /// of SO(2).
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar log() const {
    using std::atan2;
    return atan2(unitComplex().y(), unitComplex().x());
  }

  /// It re-normalizes ``unit_complex`` to unit length.
  ///
  /// Note: Because of the class invariant, there is typically no need to call
  /// this function directly.
  ///
  SOPHUS_FUNC void normalize() {
    using std::hypot;
    // Avoid under/overflows for higher precision
    Scalar length = hypot(unitComplex().x(), unitComplex().y());
    FARM_CHECK(
        length >= kEpsilon<Scalar>,
        "Complex number should not be close to zero!");
    mutUnitComplex() /= length;
  }

  /// Returns 2x2 matrix representation of the instance.
  ///
  /// For SO(2), the matrix representation is an orthogonal matrix ``R`` with
  /// ``det(R)=1``, thus the so-called "rotation matrix".
  ///
  SOPHUS_FUNC [[nodiscard]] Transformation matrix() const {
    Scalar const& real = unitComplex().x();
    Scalar const& imag = unitComplex().y();
    Transformation mat_r;
    // clang-format off
    mat_r <<
      real, -imag,
      imag,  real;
    // clang-format on
    return mat_r;
  }

  /// Assignment-like operator from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC So2Base<TDerived>& operator=(
      So2Base<TOtherDerived> const& other) {
    mutUnitComplex() = other.unitComplex();
    return *this;
  }

  /// Group multiplication, which is rotation concatenation.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC So2Product<TOtherDerived> operator*(
      So2Base<TOtherDerived> const& other) const {
    using ResultT = ReturnScalar<TOtherDerived>;
    Scalar const lhs_real = unitComplex().x();
    Scalar const lhs_imag = unitComplex().y();
    typename TOtherDerived::Scalar const& rhs_real = other.unitComplex().x();
    typename TOtherDerived::Scalar const& rhs_imag = other.unitComplex().y();
    // complex multiplication
    ResultT const result_real = lhs_real * rhs_real - lhs_imag * rhs_imag;
    ResultT const result_imag = lhs_real * rhs_imag + lhs_imag * rhs_real;

    ResultT const squared_norm =
        result_real * result_real + result_imag * result_imag;
    // We can assume that the squared-norm is close to 1 since we deal with a
    // unit complex number. Due to numerical precision issues, there might
    // be a small drift after pose concatenation. Hence, we need to renormalizes
    // the complex number here.
    // Since squared-norm is close to 1, we do not need to calculate the costly
    // square-root, but can use an approximation around 1 (see
    // http://stackoverflow.com/a/12934750 for details).
    if (squared_norm != ResultT(1.0)) {
      ResultT const scale = ResultT(2.0) / (ResultT(1.0) + squared_norm);
      return So2Product<TOtherDerived>(
          result_real * scale, result_imag * scale);
    }
    return So2Product<TOtherDerived>(result_real, result_imag);
  }

  /// Group action on 2-points.
  ///
  /// This function rotates a 2 dimensional point ``p`` by the So2 element
  ///  ``bar_R_foo`` (= rotation matrix): ``p_bar = bar_R_foo * p_foo``.
  ///
  template <
      typename TPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<TPointDerived, 2>::value>::type>
  SOPHUS_FUNC PointProduct<TPointDerived> operator*(
      Eigen::MatrixBase<TPointDerived> const& p) const {
    Scalar const& real = unitComplex().x();
    Scalar const& imag = unitComplex().y();
    return PointProduct<TPointDerived>(
        real * p[0] - imag * p[1],  //
        imag * p[0] + real * p[1]);
  }

  /// Group action on homogeneous 2-points.
  ///
  /// This function rotates a homogeneous 2 dimensional point ``p`` by the So2
  /// element ``bar_R_foo`` (= rotation matrix): ``p_bar = bar_R_foo * p_foo``.
  ///
  template <
      typename THPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<THPointDerived, 3>::value>::type>
  SOPHUS_FUNC HomogeneousPointProduct<THPointDerived> operator*(
      Eigen::MatrixBase<THPointDerived> const& p) const {
    Scalar const& real = unitComplex().x();
    Scalar const& imag = unitComplex().y();
    return HomogeneousPointProduct<THPointDerived>(
        real * p[0] - imag * p[1],  //
        imag * p[0] + real * p[1],  //
        p[2]);
  }

  /// Group action on lines.
  ///
  /// This function rotates a parametrized line ``l(t) = o + t * d`` by the So2
  /// element:
  ///
  /// Both direction ``d`` and origin ``o`` are rotated as a 2 dimensional point
  ///
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), (*this) * l.direction());
  }

  /// Group action on hyper-planes.
  ///
  /// This function rotates a hyper-plane ``n.x + d = 0`` by the So2
  /// element:
  ///
  /// Normal vector ``n`` is rotated
  /// Offset ``d`` is left unchanged
  ///
  /// Note that in 2d-case hyper-planes are just another parametrization of
  /// lines
  ///
  SOPHUS_FUNC Hyperplane operator*(Hyperplane const& p) const {
    return Hyperplane((*this) * p.normal(), p.offset());
  }

  /// In-place group multiplication. This method is only valid if the return
  /// type of the multiplication is compatible with this So2's Scalar type.
  ///
  template <
      typename TOtherDerived,
      typename = typename std::enable_if<
          std::is_same<Scalar, ReturnScalar<TOtherDerived>>::value>::type>
  SOPHUS_FUNC So2Base<TDerived> operator*=(
      So2Base<TOtherDerived> const& other) {
    *static_cast<TDerived*>(this) = *this * other;
    return *this;
  }

  /// Returns derivative of  this * So2::exp(x)  wrt. x at x=0.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kNumParameters, kDoF>
  dxThisMulExpXAt0() const {
    return Eigen::Matrix<Scalar, kNumParameters, kDoF>(
        -unitComplex()[1], unitComplex()[0]);
  }

  /// Returns internal parameters of SO(2).
  ///
  /// It returns (c[0], c[1]), with c being the unit complex number.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Vector<Scalar, kNumParameters> params()
      const {
    return unitComplex();
  }

  /// Returns derivative of log(this^{-1} * x) by x at x=this.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kDoF, kNumParameters>
  dxLogThisInvTimesXAtThis() const {
    return Eigen::Matrix<Scalar, kDoF, kNumParameters>(
        -unitComplex()[1], unitComplex()[0]);
  }

  /// Takes in complex number / tuple and normalizes it.
  ///
  /// Precondition: The complex number must not be close to zero.
  ///
  SOPHUS_FUNC void setComplex(Point const& complex) {
    mutUnitComplex() = complex;
    normalize();
  }

  /// Takes in complex number / tuple and normalizes it.
  ///
  /// Precondition: The complex number must not be close to zero.
  ///
  SOPHUS_FUNC void setParam(Point const& complex) { setComplex(); }

  /// Accessor of unit quaternion.
  ///
  SOPHUS_FUNC [[nodiscard]] ComplexT const& unitComplex() const {
    return static_cast<TDerived const*>(this)->unitComplex();
  }

 private:
  /// Mutator of unit_complex is private to ensure class invariant. That is
  /// the complex number must stay close to unit length.
  ///
  SOPHUS_FUNC
  ComplexT& mutUnitComplex() {
    return static_cast<TDerived*>(this)->mutUnitComplex();
  }
};

/// So2 using  default storage; derived from So2Base.
template <class TScalar>
class So2 : public So2Base<So2<TScalar>> {
 public:
  using Base = So2Base<So2<TScalar>>;
  static int constexpr kDoF = Base::kDoF;
  static int constexpr kNumParameters = Base::kNumParameters;

  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using ComplexMember = Eigen::Matrix<Scalar, 2, 1>;

  /// ``Base`` is friend so unit_complex_nonconst can be accessed from ``Base``.
  friend class So2Base<So2<Scalar>>;

  using Base::operator=;

  /// Define copy-assignment operator explicitly. The definition of
  /// implicit copy assignment operator is deprecated in presence of a
  /// user-declared copy constructor (-Wdeprecated-copy in clang >= 13).
  SOPHUS_FUNC So2& operator=(So2 const& other) = default;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Default constructor initializes unit complex number to identity rotation.
  ///
  SOPHUS_FUNC So2() : unit_complex_(Scalar(1), Scalar(0)) {}

  /// Copy constructor
  ///
  SOPHUS_FUNC So2(So2 const& other) = default;

  /// Copy-like constructor from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC So2(So2Base<TOtherDerived> const& other)
      : unit_complex_(other.unitComplex()) {}

  /// Constructor from rotation matrix
  ///
  /// Precondition: rotation matrix need to be orthogonal with determinant of 1.
  ///
  SOPHUS_FUNC explicit So2(Transformation const& mat_r)
      : unit_complex_(
            Scalar(0.5) * (mat_r(0, 0) + mat_r(1, 1)),
            Scalar(0.5) * (mat_r(1, 0) - mat_r(0, 1))) {
    FARM_CHECK(isOrthogonal(mat_r), "R is not orthogonal:\n {}", mat_r);
    FARM_CHECK(
        mat_r.determinant() > Scalar(0),
        "det(R) is not positive: {}",
        mat_r.determinant());
  }

  /// Constructor from pair of real and imaginary number.
  ///
  /// Precondition: The pair must not be close to zero.
  ///
  SOPHUS_FUNC So2(Scalar const& real, Scalar const& imag)
      : unit_complex_(real, imag) {
    Base::normalize();
  }

  /// Constructor from 2-vector.
  ///
  /// Precondition: The vector must not be close to zero.
  ///
  template <class TD>
  SOPHUS_FUNC explicit So2(Eigen::MatrixBase<TD> const& complex)
      : unit_complex_(complex) {
    static_assert(
        std::is_same<typename TD::Scalar, Scalar>::value,
        "must be same Scalar type");
    Base::normalize();
  }

  /// Constructor from an rotation angle.
  ///
  SOPHUS_FUNC explicit So2(Scalar theta) {
    mutUnitComplex() = So2<Scalar>::exp(theta).unitComplex();
  }

  /// Accessor of unit complex number
  ///
  SOPHUS_FUNC [[nodiscard]] ComplexMember const& unitComplex() const {
    return unit_complex_;
  }

  /// Group exponential
  ///
  /// This functions takes in an element of tangent space (= rotation angle
  /// ``theta``) and returns the corresponding element of the group SO(2).
  ///
  /// To be more specific, this function computes ``expmat(hat(omega))``
  /// with ``expmat(.)`` being the matrix exponential and ``hat(.)`` being the
  /// hat()-operator of SO(2).
  ///
  SOPHUS_FUNC static So2<Scalar> exp(Tangent const& theta) {
    using std::cos;
    using std::sin;
    return So2<Scalar>(cos(theta), sin(theta));
  }

  /// Returns derivative of exp(x) wrt. x.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpX(
      Tangent const& theta) {
    using std::cos;
    using std::sin;
    return Eigen::Matrix<Scalar, kNumParameters, kDoF>(-sin(theta), cos(theta));
  }

  /// Returns derivative of exp(x) wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpXAt0() {
    return Eigen::Matrix<Scalar, kNumParameters, kDoF>(Scalar(0), Scalar(1));
  }

  /// Returns derivative of exp(x) * p wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, 2, kDoF> dxExpXTimesPointAt0(
      Point const& point) {
    return Point(-point.y(), point.x());
  }

  /// Returns derivative of ``expmat(x)`` wrt. ``x_i at x=0``, with
  /// ``expmat(.)`` being the matrix exponential.
  ///
  SOPHUS_FUNC static Transformation dxiExpmatXAt0(int /*unused*/) {
    return generator();
  }

  /// Returns the infinitesimal generators of SO(2).
  ///
  /// The infinitesimal generators of SO(2) is:
  ///
  ///     |  0 -1 |
  ///     |  1  0 |
  ///
  SOPHUS_FUNC static Transformation generator() { return hat(Scalar(1)); }

  /// hat-operator
  ///
  /// It takes in the scalar representation ``theta`` (= rotation angle) and
  /// returns the corresponding matrix representation of Lie algebra element.
  ///
  /// Formally, the hat()-operator of SO(2) is defined as
  ///
  ///   ``hat(.): R^2 -> R^{2x2},  hat(theta) = theta * G``
  ///
  /// with ``G`` being the infinitesimal generator of SO(2).
  ///
  /// The corresponding inverse is the vee()-operator, see below.
  ///
  SOPHUS_FUNC static Transformation hat(Tangent const& theta) {
    Transformation omega;
    // clang-format off
    omega <<
        Scalar(0),   -theta,
            theta, Scalar(0);
    // clang-format on
    return omega;
  }

  /// Returns closed So2 given arbitrary 2x2 matrix.
  ///
  template <class TS = Scalar>
  static SOPHUS_FUNC std::enable_if_t<std::is_floating_point<TS>::value, So2>
  fitToSo2(Transformation const& r) {
    return So2(makeRotationMatrix(r));
  }

  /// Lie bracket
  ///
  /// It returns the Lie bracket of SO(2). Since SO(2) is a commutative group,
  /// the Lie bracket is simple ``0``.
  ///
  SOPHUS_FUNC static Tangent lieBracket(
      Tangent const& /*unused*/, Tangent const& /*unused*/) {
    return Scalar(0);
  }

  /// Draw uniform sample from SO(2) manifold.
  ///
  template <class TUniformRandomBitGenerator>
  static So2 sampleUniform(TUniformRandomBitGenerator& generator) {
    static_assert(
        kIsUniformRandomBitGeneratorV<TUniformRandomBitGenerator>,
        "generator must meet the UniformRandomBitGenerator concept");
    std::uniform_real_distribution<Scalar> uniform(-kPi<Scalar>, kPi<Scalar>);
    return So2(uniform(generator));
  }

  /// vee-operator
  ///
  /// It takes the 2x2-matrix representation ``Omega`` and maps it to the
  /// corresponding scalar representation of Lie algebra.
  ///
  /// This is the inverse of the hat()-operator, see above.
  ///
  /// Precondition: ``Omega`` must have the following structure:
  ///
  ///                |  0 -a |
  ///                |  a  0 |
  ///
  SOPHUS_FUNC static Tangent vee(Transformation const& omega) {
    using std::abs;
    return omega(1, 0);
  }

 protected:
  /// Mutator of complex number is protected to ensure class invariant.
  ///
  SOPHUS_FUNC ComplexMember& mutUnitComplex() { return unit_complex_; }

  ComplexMember unit_complex_;  // NOLINT
};

}  // namespace sophus

namespace Eigen {  // NOLINT

/// Specialization of Eigen::Map for ``So2``; derived from So2Base.
///
/// Allows us to wrap So2 objects around POD array (e.g. external c style
/// complex number / tuple).
template <class TScalar>
class Map<sophus::So2<TScalar>>
    : public sophus::So2Base<Map<sophus::So2<TScalar>>> {
 public:
  using Base = sophus::So2Base<Map<sophus::So2<TScalar>>>;
  using Scalar = TScalar;

  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  /// ``Base`` is friend so unit_complex_nonconst can be accessed from ``Base``.
  friend class sophus::So2Base<Map<sophus::So2<TScalar>>>;

  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  explicit Map(Scalar* coeffs) : unit_complex_(coeffs) {}

  /// Accessor of unit complex number.
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector2<Scalar>> const& unitComplex()
      const {
    return unit_complex_;
  }

 protected:
  /// Mutator of unit_complex is protected to ensure class invariant.
  ///
  SOPHUS_FUNC
  Map<Eigen::Vector2<Scalar>>& mutUnitComplex() { return unit_complex_; }

  Map<Eigen::Matrix<Scalar, 2, 1>> unit_complex_;  // NOLINT
};

/// Specialization of Eigen::Map for ``So2 const``; derived from So2Base.
///
/// Allows us to wrap So2 objects around POD array (e.g. external c style
/// complex number / tuple).
template <class TScalar>
class Map<sophus::So2<TScalar> const>
    : public sophus::So2Base<Map<sophus::So2<TScalar> const>> {
 public:
  using Base = sophus::So2Base<Map<sophus::So2<TScalar> const>>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar const* coeffs) : unit_complex_(coeffs) {}

  /// Accessor of unit complex number.
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector2<Scalar> const> const&
  unitComplex() const {
    return unit_complex_;
  }

 protected:
  /// Mutator of unit_complex is protected to ensure class invariant.
  ///
  Map<Eigen::Matrix<Scalar, 2, 1> const> unit_complex_;  // NOLINT
};
}  // namespace Eigen
