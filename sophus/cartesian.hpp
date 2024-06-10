/// @file
/// Cartesian - Euclidean vector space as Lie group

#pragma once
#include <sophus/types.hpp>

namespace Sophus {
template <class Scalar_, int M, int Options = 0>
class Cartesian;

template <class Scalar_>
using Cartesian2 = Cartesian<Scalar_, 2>;

template <class Scalar_>
using Cartesian3 = Cartesian<Scalar_, 3>;

using Cartesian2d = Cartesian2<double>;
using Cartesian3d = Cartesian3<double>;

}  // namespace Sophus

namespace Eigen {
namespace internal {

template <class Scalar_, int M, int Options>
struct traits<Sophus::Cartesian<Scalar_, M, Options>> {
  using Scalar = Scalar_;
  using ParamsType = Sophus::Vector<Scalar, M, Options>;
};

template <class Scalar_, int M, int Options>
struct traits<Map<Sophus::Cartesian<Scalar_, M>, Options>>
    : traits<Sophus::Cartesian<Scalar_, M, Options>> {
  using Scalar = Scalar_;
  using ParamsType = Map<Sophus::Vector<Scalar, M>, Options>;
};

template <class Scalar_, int M, int Options>
struct traits<Map<Sophus::Cartesian<Scalar_, M> const, Options>>
    : traits<Sophus::Cartesian<Scalar_, M, Options> const> {
  using Scalar = Scalar_;
  using ParamsType = Map<Sophus::Vector<Scalar, M> const, Options>;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {

/// Cartesian base type - implements Cartesian class but is storage agnostic.
///
/// Euclidean vector space as Lie group.
///
/// Lie groups can be seen as a generalization over the Euclidean vector
/// space R^M. Here a N-dimensional vector ``p`` is represented as a
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
///  - to be used in templated/generic algorithms (such as Sophus::Spline)
///    which are implemented against the Lie group interface.
///
/// Obviously, Cartesian(M) can just be represented as a M-tuple.
///
/// Cartesian is not compact, but a commutative group. For vector additions it
/// holds `a+b = b+a`.
///
/// See Cartesian class  for more details below.
///
template <class Derived, int M>
class CartesianBase {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using ParamsType = typename Eigen::internal::traits<Derived>::ParamsType;
  /// Degrees of freedom of manifold, equals to number of Cartesian coordinates.
  static int constexpr DoF = M;
  /// Number of internal parameters used, also M.
  static int constexpr num_parameters = M;
  /// Group transformations are (M+1)x(M+1) matrices.
  static int constexpr N = M + 1;
  static int constexpr Dim = M;

  using Transformation = Sophus::Matrix<Scalar, N, N>;
  using Point = Sophus::Vector<Scalar, M>;
  using HomogeneousPoint = Sophus::Vector<Scalar, N>;
  using Line = ParametrizedLine<Scalar, M>;
  using Hyperplane = Eigen::Hyperplane<Scalar, M>;
  using Tangent = Sophus::Vector<Scalar, DoF>;
  using Adjoint = Matrix<Scalar, DoF, DoF>;

  /// For binary operations the return type is determined with the
  /// ScalarBinaryOpTraits feature of Eigen. This allows mixing concrete and Map
  /// types, as well as other compatible scalar types such as Ceres::Jet and
  /// double scalars with Cartesian operations.
  template <typename OtherDerived>
  using ReturnScalar = typename Eigen::ScalarBinaryOpTraits<
      Scalar, typename OtherDerived::Scalar>::ReturnType;

  template <typename OtherDerived>
  using CartesianSum = Cartesian<ReturnScalar<OtherDerived>, M>;

  template <typename PointDerived>
  using PointProduct = Sophus::Vector<ReturnScalar<PointDerived>, M>;

  template <typename HPointDerived>
  using HomogeneousPointProduct =
      Sophus::Vector<ReturnScalar<HPointDerived>, N>;

  /// Adjoint transformation
  ///
  /// Always identity of commutative groups.
  SOPHUS_FUNC Adjoint Adj() const { return Adjoint::Identity(); }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class NewScalarType>
  SOPHUS_FUNC Cartesian<NewScalarType, M> cast() const {
    return Cartesian<NewScalarType, M>(params().template cast<NewScalarType>());
  }

  /// Returns derivative of  this * exp(x)  wrt x at x=0.
  ///
  SOPHUS_FUNC Matrix<Scalar, num_parameters, DoF> Dx_this_mul_exp_x_at_0()
      const {
    Sophus::Matrix<Scalar, num_parameters, DoF> m;
    m.setIdentity();
    return m;
  }

  /// Returns derivative of log(this^{-1} * x) by x at x=this.
  ///
  SOPHUS_FUNC Matrix<Scalar, num_parameters, DoF> Dx_log_this_inv_by_x_at_this()
      const {
    Matrix<Scalar, DoF, num_parameters> m;
    m.setIdentity();
    return m;
  }

  /// Returns group inverse.
  ///
  /// The additive inverse.
  ///
  SOPHUS_FUNC Cartesian<Scalar, M> inverse() const {
    return Cartesian<Scalar, M>(-params());
  }

  /// Logarithmic map
  ///
  /// For Euclidean vector space, just the identity. Or to be more precise
  /// it just extracts the significant M-vector from the NxN matrix.
  ///
  SOPHUS_FUNC Tangent log() const { return params(); }

  /// Returns 4x4 matrix representation of the instance.
  ///
  /// It has the following form:
  ///
  ///   | I p |
  ///   | o 1 |
  ///
  SOPHUS_FUNC Transformation matrix() const {
    Sophus::Matrix<Scalar, N, N> matrix;
    matrix.setIdentity();
    matrix.col(M).template head<M>() = params();
    return matrix;
  }

  /// Group multiplication, are vector additions.
  ///
  template <class OtherDerived>
  SOPHUS_FUNC CartesianBase<Derived, M>& operator=(
      CartesianBase<OtherDerived, M> const& other) {
    params() = other.params();
    return *this;
  }

  /// Group multiplication, are vector additions.
  ///
  template <typename OtherDerived>
  SOPHUS_FUNC CartesianSum<OtherDerived> operator*(
      CartesianBase<OtherDerived, M> const& other) const {
    return CartesianSum<OtherDerived>(params() + other.params());
  }

  /// Group action on points, again just vector addition.
  ///
  template <typename PointDerived,
            typename = typename std::enable_if_t<
                IsFixedSizeVector<PointDerived, M>::value>>
  SOPHUS_FUNC PointProduct<PointDerived> operator*(
      Eigen::MatrixBase<PointDerived> const& p) const {
    return PointProduct<PointDerived>(params() + p);
  }

  /// Group action on homogeneous points. See above for more details.
  ///
  template <typename HPointDerived,
            typename = typename std::enable_if_t<
                IsFixedSizeVector<HPointDerived, N>::value>>
  SOPHUS_FUNC HomogeneousPointProduct<HPointDerived> operator*(
      Eigen::MatrixBase<HPointDerived> const& p) const {
    const auto rp = *this * p.template head<M>();
    HomogeneousPointProduct<HPointDerived> r;
    r << rp, p(M);
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
  template <typename OtherDerived,
            typename = typename std::enable_if_t<
                std::is_same<Scalar, ReturnScalar<OtherDerived>>::value>>
  SOPHUS_FUNC CartesianBase<Derived, M>& operator*=(
      CartesianBase<OtherDerived, M> const& other) {
    *static_cast<Derived*>(this) = *this * other;
    return *this;
  }

  /// Mutator of params vector.
  ///
  SOPHUS_FUNC ParamsType& params() {
    return static_cast<Derived*>(this)->params();
  }

  /// Accessor of params vector
  ///
  SOPHUS_FUNC ParamsType const& params() const {
    return static_cast<Derived const*>(this)->params();
  }
};

/// Cartesian using default storage; derived from CartesianBase.
template <class Scalar_, int M, int Options>
class Cartesian : public CartesianBase<Cartesian<Scalar_, M, Options>, M> {
  using Base = CartesianBase<Cartesian<Scalar_, M, Options>, M>;

 public:
  static int constexpr DoF = Base::DoF;
  static int constexpr num_parameters = Base::num_parameters;
  static int constexpr N = Base::N;
  static int constexpr Dim = Base::Dim;

  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using ParamsMember = Sophus::Vector<Scalar, M, Options>;

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
  template <class OtherDerived>
  SOPHUS_FUNC Cartesian(CartesianBase<OtherDerived, M> const& other)
      : params_(other.params()) {
    static_assert(std::is_same<typename OtherDerived::Scalar, Scalar>::value,
                  "must be same Scalar type");
  }

  /// Accepts either M-vector or (M+1)x(M+1) matrices.
  ///
  template <class D>
  explicit SOPHUS_FUNC Cartesian(Eigen::MatrixBase<D> const& m) {
    static_assert(
        std::is_same<typename Eigen::MatrixBase<D>::Scalar, Scalar>::value, "");
    if (m.rows() == DoF && m.cols() == 1) {
      // trick so this compiles
      params_ = m.template block<M, 1>(0, 0);
    } else if (m.rows() == N && m.cols() == N) {
      params_ = m.template block<M, 1>(0, M);
    } else {
      SOPHUS_ENSURE(false, "{} {}", m.rows(), m.cols());
    }
  }

  /// This provides unsafe read/write access to internal data.
  ///
  SOPHUS_FUNC Scalar* data() { return params_.data(); }

  /// Const version of data() above.
  ///
  SOPHUS_FUNC Scalar const* data() const { return params_.data(); }

  /// Returns derivative of exp(x) wrt. x.
  ///
  SOPHUS_FUNC static Sophus::Matrix<Scalar, num_parameters, DoF>
  Dx_exp_x_at_0() {
    Sophus::Matrix<Scalar, num_parameters, DoF> m;
    m.setIdentity();
    return m;
  }

  /// Returns derivative of exp(x) wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Sophus::Matrix<Scalar, num_parameters, DoF> Dx_exp_x(
      Tangent const&) {
    return Dx_exp_x_at_0();
  }

  /// Returns derivative of exp(x) * p wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Sophus::Matrix<Scalar, Dim, DoF> Dx_exp_x_times_point_at_0(
      Point const&) {
    Sophus::Matrix<Scalar, Dim, DoF> J;
    J.setIdentity();
    return J;
  }

  /// Returns derivative of exp(x).matrix() wrt. ``x_i at x=0``.
  ///
  SOPHUS_FUNC static Transformation Dxi_exp_x_matrix_at_0(int i) {
    return generator(i);
  }

  /// Mutator of params vector
  ///
  SOPHUS_FUNC ParamsMember& params() { return params_; }

  /// Accessor of params vector
  ///
  SOPHUS_FUNC ParamsMember const& params() const { return params_; }

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
    SOPHUS_ENSURE(i >= 0 && i <= M, "i should be in range [0,M-1].");
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
  SOPHUS_FUNC static Cartesian<Scalar, M> exp(Tangent const& a) {
    return Cartesian<Scalar, M>(a);
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
    Transformation Omega;
    Omega.setZero();
    Omega.col(M).template head<M>() = a.template head<M>();
    return Omega;
  }

  /// Lie bracket
  ///
  /// Always 0 for commutative groups.
  SOPHUS_FUNC static Tangent lieBracket(Tangent const&, Tangent const&) {
    return Tangent::Zero();
  }

  /// Draws uniform samples in the range [-1, 1] per coordinates.
  ///
  template <class UniformRandomBitGenerator>
  static Cartesian sampleUniform(UniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    Vector<Scalar, M> v;
    for (int i = 0; i < M; ++i) {
      v[i] = uniform(generator);
    }
    return Cartesian(v);
  }

  /// vee-operator
  ///
  /// This is the inverse of the hat()-operator, see above.
  ///
  SOPHUS_FUNC static Tangent vee(Transformation const& m) {
    return m.col(M).template head<M>();
  }

 protected:
  ParamsMember params_;
};

}  // namespace Sophus

namespace Eigen {

/// Specialization of Eigen::Map for ``Cartesian``; derived from
/// CartesianBase.
///
/// Allows us to wrap Cartesian objects around POD array.
template <class Scalar_, int M, int Options>
class Map<Sophus::Cartesian<Scalar_, M>, Options>
    : public Sophus::CartesianBase<Map<Sophus::Cartesian<Scalar_, M>, Options>,
                                   M> {
 public:
  using Base =
      Sophus::CartesianBase<Map<Sophus::Cartesian<Scalar_, M>, Options>, M>;
  using Scalar = Scalar_;
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
  SOPHUS_FUNC Map<Sophus::Vector<Scalar, M, Options>>& params() {
    return params_;
  }

  /// Accessor of params vector
  ///
  SOPHUS_FUNC Map<Sophus::Vector<Scalar, M, Options>> const& params() const {
    return params_;
  }

 protected:
  Map<Sophus::Vector<Scalar, M>, Options> params_;
};

/// Specialization of Eigen::Map for ``Cartesian const``; derived from
/// CartesianBase.
///
/// Allows us to wrap Cartesian objects around POD array.
template <class Scalar_, int M, int Options>
class Map<Sophus::Cartesian<Scalar_, M> const, Options>
    : public Sophus::CartesianBase<
          Map<Sophus::Cartesian<Scalar_, M> const, Options>, M> {
 public:
  using Base =
      Sophus::CartesianBase<Map<Sophus::Cartesian<Scalar_, M> const, Options>,
                            M>;
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;

  using Base::operator*;

  SOPHUS_FUNC Map(Scalar const* coeffs) : params_(coeffs) {}

  /// Accessor of params vector
  ///
  SOPHUS_FUNC Map<Sophus::Vector<Scalar, M> const, Options> const& params()
      const {
    return params_;
  }

 protected:
  Map<Sophus::Vector<Scalar, M> const, Options> const params_;
};
}  // namespace Eigen
