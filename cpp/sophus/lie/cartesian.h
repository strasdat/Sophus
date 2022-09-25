// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Cartesian - Euclidean vector space as Lie group

#pragma once
#include "sophus/common/types.h"

namespace sophus {
template <class TScalar, int kM, int kOptions = 0>
class Cartesian;

template <class TScalar>
using Cartesian2 = Cartesian<TScalar, 2>;

template <class TScalar>
using Cartesian3 = Cartesian<TScalar, 3>;

using Cartesian2F64 = Cartesian2<double>;
using Cartesian3F64 = Cartesian3<double>;

/* [[deprecated]] */ using Cartesian2d = Cartesian2F64;
/* [[deprecated]] */ using Cartesian3d = Cartesian3F64;

}  // namespace sophus

namespace Eigen {  // NOLINT
namespace internal {

template <class TScalar, int kM, int kOptions>
struct traits<sophus::Cartesian<TScalar, kM, kOptions>> {
  using Scalar = TScalar;
  using ParamsType = Eigen::Matrix<Scalar, kM, 1, kOptions>;
};

template <class TScalar, int kM, int kOptions>
struct traits<Map<sophus::Cartesian<TScalar, kM>, kOptions>>
    : traits<sophus::Cartesian<TScalar, kM, kOptions>> {
  using Scalar = TScalar;
  using ParamsType = Map<Eigen::Vector<Scalar, kM>, kOptions>;
};

template <class TScalar, int kM, int kOptions>
struct traits<Map<sophus::Cartesian<TScalar, kM> const, kOptions>>
    : traits<sophus::Cartesian<TScalar, kM, kOptions> const> {
  using Scalar = TScalar;
  using ParamsType = Map<Eigen::Vector<Scalar, kM> const, kOptions>;
};
}  // namespace internal
}  // namespace Eigen

namespace sophus {

/// Cartesian base type - implements Cartesian class but is storage agnostic.
///
/// Euclidean vector space as Lie group.
///
/// Lie groups can be seen as a generalization over the Euclidean vector
/// space R^M. Here a kMatrixDim-dimensional vector ``p`` is represented as a
//  (M+1) x (M+1) homogeneous matrix:
///
///   | I p |
///   | o 1 |
///
/// On the other hand, Cartesian(M) can be seen as a special case of SE(M)
/// with identity rotation, and hence represents pure translation.
///
/// The purpose of this class is two-fold:
///  - for educational purpose, to highlight how Lie groups generalize over
///    Euclidean vector spaces.
///  - to be used in templated/generic algorithms (such as sophus::Spline)
///    which are implemented against the Lie group interface.
///
/// Obviously, Cartesian(M) can just be represented as a M-tuple.
///
/// Cartesian is not compact, but a commutative group. For vector additions it
/// holds `a+b = b+a`.
///
/// See Cartesian class  for more details below.
///
template <class TDerived, int kM>
class CartesianBase {
 public:
  using Scalar = typename Eigen::internal::traits<TDerived>::Scalar;
  using ParamsType = typename Eigen::internal::traits<TDerived>::ParamsType;
  /// Degrees of freedom of manifold, equals to number of Cartesian coordinates.
  static int constexpr kDoF = kM;
  /// Number of internal parameters used, also M.
  static int constexpr kNumParameters = kM;
  /// Group transformations are (M+1)x(M+1) matrices.
  static int constexpr kMatrixDim = kM + 1;
  static int constexpr kPointDim = kM;

  using Transformation = Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim>;
  using Point = Eigen::Vector<Scalar, kM>;
  using HomogeneousPoint = Eigen::Vector<Scalar, kMatrixDim>;
  using Line = Eigen::ParametrizedLine<Scalar, kM>;
  using Hyperplane = Eigen::Hyperplane<Scalar, kM>;
  using Tangent = Eigen::Vector<Scalar, kDoF>;
  using Adjoint = Eigen::Matrix<Scalar, kDoF, kDoF>;

  /// For binary operations the return type is determined with the
  /// ScalarBinaryOpTraits feature of Eigen. This allows mixing concrete and Map
  /// types, as well as other compatible scalar types such as Ceres::Jet and
  /// double scalars with Cartesian operations.
  template <class TOtherDerived>
  using ReturnScalar = typename Eigen::
      ScalarBinaryOpTraits<Scalar, typename TOtherDerived::Scalar>::ReturnType;

  template <class TOtherDerived>
  using CartesianSum = Cartesian<ReturnScalar<TOtherDerived>, kM>;

  template <class TPointDerived>
  using PointProduct = Eigen::Vector<ReturnScalar<TPointDerived>, kM>;

  template <class THPointDerived>
  using HomogeneousPointProduct =
      Eigen::Vector<ReturnScalar<THPointDerived>, kMatrixDim>;

  /// Adjoint transformation
  ///
  /// Always identity of commutative groups.
  SOPHUS_FUNC [[nodiscard]] Adjoint adj() const { return Adjoint::Identity(); }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class TNewScalarType>
  SOPHUS_FUNC Cartesian<TNewScalarType, kM> cast() const {
    return Cartesian<TNewScalarType, kM>(
        params().template cast<TNewScalarType>());
  }

  /// Returns derivative of  this * exp(x)  wrt x at x=0.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kNumParameters, kDoF>
  dxThisMulExpXAt0() const {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> m;
    m.setIdentity();
    return m;
  }

  /// Returns derivative of log(this^{-1} * x) by x at x=this.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kNumParameters, kDoF>
  dxLogThisInvTimesXAtThis() const {
    Eigen::Matrix<Scalar, kDoF, kNumParameters> m;
    m.setIdentity();
    return m;
  }

  /// Returns group inverse.
  ///
  /// The additive inverse.
  ///
  SOPHUS_FUNC [[nodiscard]] Cartesian<Scalar, kM> inverse() const {
    return Cartesian<Scalar, kM>(-params());
  }

  /// Logarithmic map
  ///
  /// For Euclidean vector space, just the identity. Or to be more precise
  /// it just extracts the significant M-vector from the NxN matrix.
  ///
  SOPHUS_FUNC [[nodiscard]] Tangent log() const { return params(); }

  /// Returns 4x4 matrix representation of the instance.
  ///
  /// It has the following form:
  ///
  ///   | I p |
  ///   | o 1 |
  ///
  SOPHUS_FUNC [[nodiscard]] Transformation matrix() const {
    Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim> matrix;
    matrix.setIdentity();
    matrix.col(kM).template head<kM>() = params();
    return matrix;
  }

  /// Group multiplication, are vector additions.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC CartesianBase<TDerived, kM>& operator=(
      CartesianBase<TOtherDerived, kM> const& other) {
    params() = other.params();
    return *this;
  }

  /// Group multiplication, are vector additions.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC CartesianSum<TOtherDerived> operator*(
      CartesianBase<TOtherDerived, kM> const& other) const {
    return CartesianSum<TOtherDerived>(params() + other.params());
  }

  /// Group action on points, again just vector addition.
  ///
  template <
      typename TPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<TPointDerived, kM>::value>::type>
  SOPHUS_FUNC PointProduct<TPointDerived> operator*(
      Eigen::MatrixBase<TPointDerived> const& p) const {
    return PointProduct<TPointDerived>(params() + p);
  }

  /// Group action on homogeneous points. See above for more details.
  ///
  template <
      typename THPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<THPointDerived, kMatrixDim>::value>::type>
  SOPHUS_FUNC HomogeneousPointProduct<THPointDerived> operator*(
      Eigen::MatrixBase<THPointDerived> const& p) const {
    auto const rp = *this * p.template head<kM>();
    HomogeneousPointProduct<THPointDerived> r;
    r << rp, p(kM);
    return r;
  }

  /// Group action on lines.
  ///
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), l.direction());
  }

  /// Group action on planes.
  ///
  SOPHUS_FUNC Hyperplane operator*(Hyperplane const& p) const {
    return Hyperplane(p.normal(), p.offset() - params().dot(p.normal()));
  }

  /// In-place group multiplication. This method is only valid if the return
  /// type of the multiplication is compatible with this Cartesian's Scalar
  /// type.
  ///
  template <
      typename TOtherDerived,
      typename = typename std::enable_if<
          std::is_same<Scalar, ReturnScalar<TOtherDerived>>::value>::type>
  SOPHUS_FUNC CartesianBase<TDerived, kM>& operator*=(
      CartesianBase<TOtherDerived, kM> const& other) {
    *static_cast<TDerived*>(this) = *this * other;
    return *this;
  }

  /// Mutator of params vector.
  ///
  SOPHUS_FUNC ParamsType& params() {
    return static_cast<TDerived*>(this)->params();
  }

  /// Accessor of params vector
  ///
  SOPHUS_FUNC [[nodiscard]] ParamsType const& params() const {
    return static_cast<TDerived const*>(this)->params();
  }
};

/// Cartesian using default storage; derived from CartesianBase.
template <class TScalar, int kM, int kOptions>
class Cartesian : public CartesianBase<Cartesian<TScalar, kM, kOptions>, kM> {
  using Base = CartesianBase<Cartesian<TScalar, kM, kOptions>, kM>;

 public:
  static int constexpr kDoF = Base::kDoF;
  static int constexpr kNumParameters = Base::kNumParameters;
  static int constexpr kMatrixDim = Base::kMatrixDim;
  static int constexpr kPointDim = Base::kPointDim;

  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using ParamsMember = Eigen::Matrix<Scalar, kM, 1, kOptions>;

  using Base::operator=;

  /// Define copy-assignment operator explicitly. The definition of
  /// implicit copy assignment operator is deprecated in presence of a
  /// user-declared copy constructor (-Wdeprecated-copy in clang >= 13).
  SOPHUS_FUNC Cartesian& operator=(Cartesian const& other) = default;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Default constructor initializes to zero vector.
  ///
  SOPHUS_FUNC Cartesian() { params_.setZero(); }

  /// Copy constructor
  ///
  SOPHUS_FUNC Cartesian(Cartesian const& other) = default;

  /// Copy-like constructor from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Cartesian(CartesianBase<TOtherDerived, kM> const& other)
      : params_(other.params()) {
    static_assert(
        std::is_same<typename TOtherDerived::Scalar, Scalar>::value,
        "must be same Scalar type");
  }

  /// Accepts either M-vector or (M+1)x(M+1) matrices.
  ///
  template <class TD>
  explicit SOPHUS_FUNC Cartesian(Eigen::MatrixBase<TD> const& m) {
    static_assert(
        std::is_same<typename Eigen::MatrixBase<TD>::Scalar, Scalar>::value,
        "");
    if (m.rows() == kDoF && m.cols() == 1) {
      // trick so this compiles
      params_ = m.template block<kM, 1>(0, 0);
    } else if (m.rows() == kMatrixDim && m.cols() == kMatrixDim) {
      params_ = m.template block<kM, 1>(0, kM);
    } else {
      FARM_CHECK(false, "{} {}", m.rows(), m.cols());
    }
  }

  /// This provides unsafe read/write access to internal data.
  ///
  SOPHUS_FUNC Scalar* data() { return params_.data(); }

  /// Const version of data() above.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar const* data() const {
    return params_.data();
  }

  /// Returns derivative of exp(x) wrt. x.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpXAt0() {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> m;
    m.setIdentity();
    return m;
  }

  /// Returns derivative of exp(x) wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpX(
      Tangent const& /*unused*/) {
    return dxExpXAt0();
  }

  /// Returns derivative of exp(x) * p wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kPointDim, kDoF> dxExpXTimesPointAt0(
      Point const& /*unused*/) {
    Eigen::Matrix<Scalar, kPointDim, kDoF> j;
    j.setIdentity();
    return j;
  }

  /// Returns derivative of ``expmat(x)`` wrt. ``x_i at x=0``, with
  /// ``expmat(.)`` being the matrix exponential.
  ///
  SOPHUS_FUNC static Transformation dxiExpmatXAt0(int i) {
    return generator(i);
  }

  /// Mutator of params vector
  ///
  SOPHUS_FUNC ParamsMember& params() { return params_; }

  /// Accessor of params vector
  ///
  SOPHUS_FUNC [[nodiscard]] ParamsMember const& params() const {
    return params_;
  }

  /// Returns the ith infinitesimal generators of Cartesian(M).
  ///
  /// The infinitesimal generators for e.g. the 3-dimensional case:
  ///
  /// ```
  ///         |  0  0  0  1 |
  ///   G_0 = |  0  0  0  0 |
  ///         |  0  0  0  0 |
  ///         |  0  0  0  0 |
  ///
  ///         |  0  0  0  0 |
  ///   G_1 = |  0  0  0  1 |
  ///         |  0  0  0  0 |
  ///         |  0  0  0  0 |
  ///
  ///         |  0  0  0  0 |
  ///   G_2 = |  0  0  0  0 |
  ///         |  0  0  0  1 |
  ///         |  0  0  0  0 |
  /// ```
  ///
  /// Precondition: ``i`` must be in [0, M-1].
  ///
  SOPHUS_FUNC static Transformation generator(int i) {
    FARM_CHECK(i >= 0 && i <= kM, "i should be in range [0,M-1].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  /// Group exponential
  ///
  /// For Euclidean vector space, just the identity. Or to be more precise
  /// it just constructs the (M+1xM+1) homogeneous matrix representation
  //  from the M-vector.
  ///
  SOPHUS_FUNC static Cartesian<Scalar, kM> exp(Tangent const& a) {
    return Cartesian<Scalar, kM>(a);
  }

  /// hat-operator
  ///
  /// Formally, the hat()-operator of Cartesian(M) is defined as
  ///
  ///   ``hat(.): R^M -> R^{M+1xM+1},  hat(a) = sum_i a_i * G_i``
  ///   (for i=0,...,M-1)
  ///
  /// with ``G_i`` being the ith infinitesimal generator of Cartesian(M).
  ///
  /// The corresponding inverse is the vee()-operator, see below.
  ///
  SOPHUS_FUNC static Transformation hat(Tangent const& a) {
    Transformation omega;
    omega.setZero();
    omega.col(kM).template head<kM>() = a.template head<kM>();
    return omega;
  }

  /// Lie bracket
  ///
  /// Always 0 for commutative groups.
  SOPHUS_FUNC static Tangent lieBracket(
      Tangent const& /*unused*/, Tangent const& /*unused*/) {
    return Tangent::Zero();
  }

  /// Draws uniform samples in the range [-1, 1] per coordinates.
  ///
  template <class TUniformRandomBitGenerator>
  static Cartesian sampleUniform(TUniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    Eigen::Vector<Scalar, kM> v;
    for (int i = 0; i < kM; ++i) {
      v[i] = uniform(generator);
    }
    return Cartesian(v);
  }

  /// vee-operator
  ///
  /// This is the inverse of the hat()-operator, see above.
  ///
  SOPHUS_FUNC static Tangent vee(Transformation const& m) {
    return m.col(kM).template head<kM>();
  }

 protected:
  ParamsMember params_;  // NOLINT
};

}  // namespace sophus

namespace Eigen {  // NOLINT

/// Specialization of Eigen::Map for ``Cartesian``; derived from
/// CartesianBase.
///
/// Allows us to wrap Cartesian objects around POD array.
template <class TScalar, int kM, int kOptions>
class Map<sophus::Cartesian<TScalar, kM>, kOptions>
    : public sophus::
          CartesianBase<Map<sophus::Cartesian<TScalar, kM>, kOptions>, kM> {
 public:
  using Base =
      sophus::CartesianBase<Map<sophus::Cartesian<TScalar, kM>, kOptions>, kM>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;

  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar* coeffs) : params_(coeffs) {}

  /// Mutator of params vector
  ///
  SOPHUS_FUNC Map<Eigen::Vector<Scalar, kM>, kOptions>& params() {
    return params_;
  }

  /// Accessor of params vector
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector<Scalar, kM>, kOptions> const&
  params() const {
    return params_;
  }

 protected:
  Map<Eigen::Vector<Scalar, kM>, kOptions> params_;  // NOLINT
};

/// Specialization of Eigen::Map for ``Cartesian const``; derived from
/// CartesianBase.
///
/// Allows us to wrap Cartesian objects around POD array.
template <class TScalar, int kM, int kOptions>
class Map<sophus::Cartesian<TScalar, kM> const, kOptions>
    : public sophus::CartesianBase<
          Map<sophus::Cartesian<TScalar, kM> const, kOptions>,
          kM> {
 public:
  using Base = sophus::
      CartesianBase<Map<sophus::Cartesian<TScalar, kM> const, kOptions>, kM>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;

  using Base::operator*;

  SOPHUS_FUNC Map(Scalar const* coeffs) : params_(coeffs) {}

  /// Accessor of params vector
  ///
  SOPHUS_FUNC
  [[nodiscard]] Map<Eigen::Vector<Scalar, kM> const, kOptions> const& params()
      const {
    return params_;
  }

 protected:
  Map<Eigen::Vector<Scalar, kM> const, kOptions> params_;  // NOLINT
};
}  // namespace Eigen
