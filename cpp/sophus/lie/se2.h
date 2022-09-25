// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Special Euclidean group SE(2) - rotation and translation in 2d.

#pragma once

#include "sophus/lie/so2.h"

namespace sophus {
template <class TScalar, int kOptions = 0>
class Se2;
using Se2F64 = Se2<double>;
using Se2F32 = Se2<float>;

template <class TScalar, int kOptions = 0>
/* [[deprecated]] */ using SE2 = Se2<TScalar, kOptions>;
/* [[deprecated]] */ using SE2d = Se2F64;
/* [[deprecated]] */ using SE2f = Se2F32;

}  // namespace sophus

namespace Eigen {  // NOLINT
namespace internal {

template <class TScalar, int kOptions>
struct traits<sophus::Se2<TScalar, kOptions>> {
  using Scalar = TScalar;
  using TranslationType = Eigen::Matrix<Scalar, 2, 1, kOptions>;
  using So2Type = sophus::So2<Scalar, kOptions>;
};

template <class TScalar, int kOptions>
struct traits<Map<sophus::Se2<TScalar>, kOptions>>
    : traits<sophus::Se2<TScalar, kOptions>> {
  using Scalar = TScalar;
  using TranslationType = Map<Eigen::Vector2<Scalar>, kOptions>;
  using So2Type = Map<sophus::So2<Scalar>, kOptions>;
};

template <class TScalar, int kOptions>
struct traits<Map<sophus::Se2<TScalar> const, kOptions>>
    : traits<sophus::Se2<TScalar, kOptions> const> {
  using Scalar = TScalar;
  using TranslationType = Map<Eigen::Vector2<Scalar> const, kOptions>;
  using So2Type = Map<sophus::So2<Scalar> const, kOptions>;
};
}  // namespace internal
}  // namespace Eigen

namespace sophus {

/// Se2 base type - implements Se2 class but is storage agnostic.
///
/// SE(2) is the group of rotations  and translation in 2d. It is the
/// semi-direct product of SO(2) and the 2d Euclidean vector space.  The class
/// is represented using a composition of SO2Group  for rotation and a 2-vector
/// for translation.
///
/// SE(2) is neither compact, nor a commutative group.
///
///  - 3x3 Eigen::Matrix representation:
///
/// ```
///   | R t |
///   | o 1 |
/// ```
///
/// where ``R`` is a 2x2 rotation matrix, ``t`` a translation 2-vector and
/// ``o`` a 2-column vector of zeros.
///
///  - Tangent 3-vector: [upsilon, theta],
///
/// where ``upsilon`` is the translational velocity 2-vector and ``theta`` the
/// rotational velocity.
///
///  - Internal 4-vector representation: [t0, t1, zr, zi],
///
/// with ``t0, t1`` are the translational components, and ``zr``
/// the real part and ``zi`` the imaginary part of a unit length complex number.
///
/// See SO2Group for more details of the rotation representation in 2d.
///
template <class TDerived>
class Se2Base {
 public:
  using Scalar = typename Eigen::internal::traits<TDerived>::Scalar;
  using TranslationType =
      typename Eigen::internal::traits<TDerived>::TranslationType;
  using So2Type = typename Eigen::internal::traits<TDerived>::So2Type;

  /// Degrees of freedom of manifold, number of dimensions in tangent space
  /// (two for translation, three for rotation).
  static int constexpr kDoF = 3;
  /// Number of internal parameters used (tuple for complex, two for
  /// translation).
  static int constexpr kNumParameters = 4;
  /// Group transformations are 3x3 matrices.
  static int constexpr kMatrixDim = 3;
  /// Points are 2-dimensional
  static int constexpr kPointDim = 2;
  using Transformation = Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim>;
  using Point = Eigen::Vector2<Scalar>;
  using HomogeneousPoint = Eigen::Vector3<Scalar>;
  using Line = Eigen::ParametrizedLine<Scalar, 2>;
  using Hyperplane = Eigen::Hyperplane<Scalar, 2>;
  using Tangent = Eigen::Vector<Scalar, kDoF>;
  using Adjoint = Eigen::Matrix<Scalar, kDoF, kDoF>;

  /// For binary operations the return type is determined with the
  /// ScalarBinaryOpTraits feature of Eigen. This allows mixing concrete and Map
  /// types, as well as other compatible scalar types such as Ceres::Jet and
  /// double scalars with Se2 operations.
  template <class TOtherDerived>
  using ReturnScalar = typename Eigen::
      ScalarBinaryOpTraits<Scalar, typename TOtherDerived::Scalar>::ReturnType;

  template <class TOtherDerived>
  using Se2Product = Se2<ReturnScalar<TOtherDerived>>;

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
  SOPHUS_FUNC [[nodiscard]] Adjoint adj() const {
    Eigen::Matrix<Scalar, 2, 2> const& mat_r = so2().matrix();
    Transformation res;
    res.setIdentity();
    res.template topLeftCorner<2, 2>() = mat_r;
    res(0, 2) = translation()[1];
    res(1, 2) = -translation()[0];
    return res;
  }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class TNewScalarType>
  SOPHUS_FUNC [[nodiscard]] Se2<TNewScalarType> cast() const {
    return Se2<TNewScalarType>(
        so2().template cast<TNewScalarType>(),
        translation().template cast<TNewScalarType>());
  }

  /// Returns derivative of  this * exp(x)  wrt x at x=0.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kNumParameters, kDoF>
  dxThisMulExpXAt0() const {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> dx;
    Eigen::Vector2<Scalar> const z = unitComplex();
    Scalar o(0);
    // clang-format off
    dx(0, 0) = o;     dx(1, 0) = o;    dx(2, 0) =  z[0];  dx(3, 0) = z[1];
    dx(0, 1) = o;     dx(1, 1) = o;    dx(2, 1) = -z[1];  dx(3, 1) = z[0];
    dx(0, 2) = -z[1]; dx(1, 2) = z[0]; dx(2, 2) =  o;     dx(3, 2) = o;
    // clang-format on
    return dx;
  }

  /// Returns derivative of log(this^{-1} * x) by x at x=this.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kDoF, kNumParameters>
  dxLogThisInvTimesXAtThis() const {
    Eigen::Matrix<Scalar, kDoF, kNumParameters> d;
    d.template block<2, 2>(0, 0).setZero();
    d.template block<2, 2>(0, 2) = so2().inverse().matrix();
    d.template block<1, 2>(2, 0) = so2().dxLogThisInvTimesXAtThis();
    d.template block<1, 2>(2, 2).setZero();
    return d;
  }

  /// Returns group inverse.
  ///
  SOPHUS_FUNC [[nodiscard]] Se2<Scalar> inverse() const {
    So2<Scalar> const inv = so2().inverse();
    return Se2<Scalar>(inv, inv * (translation() * Scalar(-1)));
  }

  /// Logarithmic map
  ///
  /// Computes the logarithm, the inverse of the group exponential which maps
  /// element of the group (rigid body transformations) to elements of the
  /// tangent space (twist).
  ///
  /// To be specific, this function computes ``vee(logmat(.))`` with
  /// ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  /// of SE(2).
  ///
  SOPHUS_FUNC [[nodiscard]] Tangent log() const {
    using std::abs;

    Tangent vec_upsilon_theta;
    Scalar theta = so2().log();
    vec_upsilon_theta[2] = theta;
    Scalar halftheta = Scalar(0.5) * theta;
    Scalar halftheta_by_tan_of_halftheta;

    Eigen::Vector2<Scalar> z = so2().unitComplex();
    Scalar real_minus_one = z.x() - Scalar(1.);
    if (abs(real_minus_one) < kEpsilon<Scalar>) {
      halftheta_by_tan_of_halftheta =
          Scalar(1.) - Scalar(1. / 12) * theta * theta;
    } else {
      halftheta_by_tan_of_halftheta = -(halftheta * z.y()) / (real_minus_one);
    }
    Eigen::Matrix<Scalar, 2, 2> v_inv;
    v_inv << halftheta_by_tan_of_halftheta, halftheta, -halftheta,
        halftheta_by_tan_of_halftheta;
    vec_upsilon_theta.template head<2>() = v_inv * translation();
    return vec_upsilon_theta;
  }

  /// Normalize So2 element
  ///
  /// It re-normalizes the So2 element.
  ///
  SOPHUS_FUNC void normalize() { so2().normalize(); }

  /// Returns 3x3 matrix representation of the instance.
  ///
  /// It has the following form:
  ///
  /// ```
  ///   | R t |
  ///   | o 1 |
  /// ```
  ///
  /// where ``R`` is a 2x2 rotation matrix, ``t`` a translation 2-vector and
  /// ``o`` a 2-column vector of zeros.
  ///
  SOPHUS_FUNC [[nodiscard]] Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.template topLeftCorner<2, 3>() = matrix2x3();
    homogenious_matrix.row(2) =
        Eigen::Matrix<Scalar, 1, 3>(Scalar(0), Scalar(0), Scalar(1));
    return homogenious_matrix;
  }

  /// Returns the significant first two rows of the matrix above.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, 2, 3> matrix2x3() const {
    Eigen::Matrix<Scalar, 2, 3> matrix;
    matrix.template topLeftCorner<2, 2>() = rotationMatrix();
    matrix.col(2) = translation();
    return matrix;
  }

  /// Assignment-like operator from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Se2Base<TDerived>& operator=(
      Se2Base<TOtherDerived> const& other) {
    so2() = other.so2();
    translation() = other.translation();
    return *this;
  }

  /// Group multiplication, which is rotation concatenation.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Se2Product<TOtherDerived> operator*(
      Se2Base<TOtherDerived> const& other) const {
    return Se2Product<TOtherDerived>(
        so2() * other.so2(), translation() + so2() * other.translation());
  }

  /// Group action on 2-points.
  ///
  /// This function rotates and translates a two dimensional point ``p`` by the
  /// SE(2) element ``bar_transform_foo = (bar_R_foo, t_bar)`` (= rigid body
  /// transformation):
  ///
  ///   ``p_bar = bar_R_foo * p_foo + t_bar``.
  ///
  template <
      typename TPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<TPointDerived, 2>::value>::type>
  SOPHUS_FUNC PointProduct<TPointDerived> operator*(
      Eigen::MatrixBase<TPointDerived> const& p) const {
    return so2() * p + translation();
  }

  /// Group action on homogeneous 2-points. See above for more details.
  ///
  template <
      typename THPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<THPointDerived, 3>::value>::type>
  SOPHUS_FUNC HomogeneousPointProduct<THPointDerived> operator*(
      Eigen::MatrixBase<THPointDerived> const& p) const {
    PointProduct<THPointDerived> const tp =
        so2() * p.template head<2>() + p(2) * translation();
    return HomogeneousPointProduct<THPointDerived>(tp(0), tp(1), p(2));
  }

  /// Group action on lines.
  ///
  /// This function rotates and translates a parametrized line
  /// ``l(t) = o + t * d`` by the SE(2) element:
  ///
  /// Origin ``o`` is rotated and translated using SE(2) action
  /// Direction ``d`` is rotated using SO(2) action
  ///
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), so2() * l.direction());
  }

  /// Group action on hyper-planes.
  ///
  /// This function rotates a hyper-plane ``n.x + d = 0`` by the Se2
  /// element:
  ///
  /// Normal vector ``n`` is rotated
  /// Offset ``d`` is adjusted for translation
  ///
  /// Note that in 2d-case hyper-planes are just another parametrization of
  /// lines
  ///
  SOPHUS_FUNC Hyperplane operator*(Hyperplane const& p) const {
    Hyperplane const rotated = so2() * p;
    return Hyperplane(
        rotated.normal(),
        rotated.offset() - translation().dot(rotated.normal()));
  }

  /// In-place group multiplication. This method is only valid if the return
  /// type of the multiplication is compatible with this So2's Scalar type.
  ///
  template <
      typename TOtherDerived,
      typename = typename std::enable_if<
          std::is_same<Scalar, ReturnScalar<TOtherDerived>>::value>::type>
  SOPHUS_FUNC Se2Base<TDerived>& operator*=(
      Se2Base<TOtherDerived> const& other) {
    *static_cast<TDerived*>(this) = *this * other;
    return *this;
  }

  /// Returns internal parameters of SE(2).
  ///
  /// It returns (c[0], c[1], t[0], t[1]),
  /// with c being the unit complex number, t the translation 3-vector.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Vector<Scalar, kNumParameters> params()
      const {
    Eigen::Vector<Scalar, kNumParameters> p;
    p << so2().params(), translation();
    return p;
  }

  /// Returns rotation matrix.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, 2, 2> rotationMatrix() const {
    return so2().matrix();
  }

  /// Takes in complex number, and normalizes it.
  ///
  /// Precondition: The complex number must not be close to zero.
  ///
  SOPHUS_FUNC void setComplex(Eigen::Vector2<Scalar> const& complex) {
    return so2().setComplex(complex);
  }

  /// Sets ``so3`` using ``rotation_matrix``.
  ///
  /// Precondition: ``R`` must be orthogonal and ``det(R)=1``.
  ///
  SOPHUS_FUNC void setRotationMatrix(Eigen::Matrix<Scalar, 2, 2> const& r) {
    FARM_CHECK(isOrthogonal(r), "R is not orthogonal:\n {}", r);
    FARM_CHECK(
        r.determinant() > Scalar(0),
        "det(R) is not positive: {}",
        r.determinant());
    typename So2Type::ComplexTemporaryType const complex(
        Scalar(0.5) * (r(0, 0) + r(1, 1)), Scalar(0.5) * (r(1, 0) - r(0, 1)));
    so2().setComplex(complex);
  }

  /// Mutator of So3 group.
  ///
  SOPHUS_FUNC
  So2Type& so2() { return static_cast<TDerived*>(this)->so2(); }

  /// Accessor of So3 group.
  ///
  SOPHUS_FUNC [[nodiscard]] So2Type const& so2() const {
    return static_cast<TDerived const*>(this)->so2();
  }

  /// Mutator of translation vector.
  ///
  SOPHUS_FUNC
  TranslationType& translation() {
    return static_cast<TDerived*>(this)->translation();
  }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] TranslationType const& translation() const {
    return static_cast<TDerived const*>(this)->translation();
  }

  /// Accessor of unit complex number.
  ///
  SOPHUS_FUNC [[nodiscard]]
  typename Eigen::internal::traits<TDerived>::So2Type::ComplexT const&
  unitComplex() const {
    return so2().unitComplex();
  }
};

/// Se2 using default storage; derived from Se2Base.
template <class TScalar, int kOptions>
class Se2 : public Se2Base<Se2<TScalar, kOptions>> {
 public:
  using Base = Se2Base<Se2<TScalar, kOptions>>;
  static int constexpr kDoF = Base::kDoF;
  static int constexpr kNumParameters = Base::kNumParameters;

  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using SO2Member = So2<Scalar, kOptions>;
  using TranslationMember = Eigen::Matrix<Scalar, 2, 1, kOptions>;

  using Base::operator=;

  /// Define copy-assignment operator explicitly. The definition of
  /// implicit copy assignment operator is deprecated in presence of a
  /// user-declared copy constructor (-Wdeprecated-copy in clang >= 13).
  SOPHUS_FUNC Se2& operator=(Se2 const& other) = default;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Default constructor initializes rigid body motion to the identity.
  ///
  SOPHUS_FUNC Se2();

  /// Copy constructor
  ///
  SOPHUS_FUNC Se2(Se2 const& other) = default;

  /// Copy-like constructor from OtherDerived
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Se2(Se2Base<TOtherDerived> const& other)
      : so2_(other.so2()), translation_(other.translation()) {
    static_assert(
        std::is_same<typename TOtherDerived::Scalar, Scalar>::value,
        "must be same Scalar type");
  }

  /// Constructor from So3 and translation vector
  ///
  template <class TOtherDerived, class TD>
  SOPHUS_FUNC Se2(
      So2Base<TOtherDerived> const& so2,
      Eigen::MatrixBase<TD> const& translation)
      : so2_(so2), translation_(translation) {
    static_assert(
        std::is_same<typename TOtherDerived::Scalar, Scalar>::value,
        "must be same Scalar type");
    static_assert(
        std::is_same<typename TD::Scalar, Scalar>::value,
        "must be same Scalar type");
  }

  /// Constructor from rotation matrix and translation vector
  ///
  /// Precondition: Rotation matrix needs to be orthogonal with determinant
  /// of 1.
  ///
  SOPHUS_FUNC
  Se2(typename So2<Scalar>::Transformation const& rotation_matrix,
      Point const& translation)
      : so2_(rotation_matrix), translation_(translation) {}

  /// Constructor from rotation angle and translation vector.
  ///
  SOPHUS_FUNC Se2(Scalar const& theta, Point const& translation)
      : so2_(theta), translation_(translation) {}

  /// Constructor from complex number and translation vector
  ///
  /// Precondition: ``complex`` must not be close to zero.
  SOPHUS_FUNC Se2(
      Eigen::Vector2<Scalar> const& complex, Point const& translation)
      : so2_(complex), translation_(translation) {}

  /// Constructor from 3x3 matrix
  ///
  /// Precondition: Rotation matrix needs to be orthogonal with determinant
  /// of 1. The last row must be ``(0, 0, 1)``.
  ///
  SOPHUS_FUNC explicit Se2(Transformation const& mat3x3)
      : so2_(mat3x3.template topLeftCorner<2, 2>().eval()),
        translation_(mat3x3.template block<2, 1>(0, 2)) {}

  /// This provides unsafe read/write access to internal data. SO(2) is
  /// represented by a complex number (two parameters). When using direct write
  /// access, the user needs to take care of that the complex number stays
  /// normalized.
  ///
  SOPHUS_FUNC Scalar* data() {
    // so2_ and translation_ are layed out sequentially with no padding
    return so2_.data();
  }

  /// Const version of data() above.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar const* data() const {
    /// so2_ and translation_ are layed out sequentially with no padding
    return so2_.data();
  }

  /// Accessor of So3
  ///
  SOPHUS_FUNC SO2Member& so2() { return so2_; }

  /// Mutator of So3
  ///
  SOPHUS_FUNC [[nodiscard]] SO2Member const& so2() const { return so2_; }

  /// Mutator of translation vector
  ///
  SOPHUS_FUNC TranslationMember& translation() { return translation_; }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] TranslationMember const& translation() const {
    return translation_;
  }

  /// Returns derivative of exp(x) wrt. x.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpX(
      Tangent const& upsilon_theta) {
    using std::abs;
    using std::cos;
    using std::pow;
    using std::sin;
    Eigen::Matrix<Scalar, kNumParameters, kDoF> dx;
    Eigen::Vector<Scalar, 2> upsilon = upsilon_theta.template head<2>();
    Scalar theta = upsilon_theta[2];

    if (abs(theta) < kEpsilon<Scalar>) {
      Scalar const o(0);
      Scalar const i(1);

      // clang-format off
      dx << o, o, o,
            o, o, i,
            i, o, -Scalar(0.5) * upsilon[1],
            o, i,  Scalar(0.5) * upsilon[0];
      // clang-format on
      return dx;
    }

    Scalar const c0 = sin(theta);
    Scalar const c1 = cos(theta);
    Scalar const c2 = 1.0 / theta;
    Scalar const c3 = c0 * c2;
    Scalar const c4 = -c1 + Scalar(1);
    Scalar const c5 = c2 * c4;
    Scalar const c6 = c1 * c2;
    Scalar const c7 = pow(theta, -2);
    Scalar const c8 = c0 * c7;
    Scalar const c9 = c4 * c7;

    Scalar const o = Scalar(0);
    dx(0, 0) = o;
    dx(0, 1) = o;
    dx(0, 2) = -c0;

    dx(1, 0) = o;
    dx(1, 1) = o;
    dx(1, 2) = c1;

    dx(2, 0) = c3;
    dx(2, 1) = -c5;
    dx(2, 2) =
        -c3 * upsilon[1] + c6 * upsilon[0] - c8 * upsilon[0] + c9 * upsilon[1];

    dx(3, 0) = c5;
    dx(3, 1) = c3;
    dx(3, 2) =
        c3 * upsilon[0] + c6 * upsilon[1] - c8 * upsilon[1] - c9 * upsilon[0];
    return dx;
  }

  /// Returns derivative of exp(x) wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpXAt0() {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> dx;
    Scalar const o(0);
    Scalar const i(1);

    // clang-format off
    dx << o, o, o,
          o, o, i,
          i, o, o,
          o, i, o;
    // clang-format on
    return dx;
  }

  /// Returns derivative of exp(x) * p wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, 2, kDoF> dxExpXTimesPointAt0(
      Point const& point) {
    Eigen::Matrix<Scalar, 2, kDoF> dx;
    dx << Eigen::Matrix2<Scalar>::Identity(),
        sophus::So2<Scalar>::dxExpXTimesPointAt0(point);
    return dx;
  }

  /// Returns derivative of ``expmat(x)`` wrt. ``x_i at x=0``, with
  /// ``expmat(.)`` being the matrix exponential.
  ///
  SOPHUS_FUNC static Transformation dxiExpmatXAt0(int i) {
    return generator(i);
  }

  /// Group exponential
  ///
  /// This functions takes in an element of tangent space (= twist ``a``) and
  /// returns the corresponding element of the group SE(2).
  ///
  /// The first two components of ``a`` represent the translational part
  /// ``upsilon`` in the tangent space of SE(2), while the last three components
  /// of ``a`` represents the rotation vector ``omega``.
  /// To be more specific, this function computes ``expmat(hat(a))`` with
  /// ``expmat(.)`` being the matrix exponential and ``hat(.)`` the hat-operator
  /// of SE(2), see below.
  ///
  SOPHUS_FUNC static Se2<Scalar> exp(Tangent const& vec_a) {
    Scalar theta = vec_a[2];
    So2<Scalar> so2 = So2<Scalar>::exp(theta);
    Scalar sin_theta_by_theta;
    Scalar one_minus_cos_theta_by_theta;
    using std::abs;

    if (abs(theta) < kEpsilon<Scalar>) {
      Scalar theta_sq = theta * theta;
      sin_theta_by_theta = Scalar(1.) - Scalar(1. / 6.) * theta_sq;
      one_minus_cos_theta_by_theta =
          Scalar(0.5) * theta - Scalar(1. / 24.) * theta * theta_sq;
    } else {
      sin_theta_by_theta = so2.unitComplex().y() / theta;
      one_minus_cos_theta_by_theta =
          (Scalar(1.) - so2.unitComplex().x()) / theta;
    }
    Eigen::Vector2<Scalar> trans(
        sin_theta_by_theta * vec_a[0] - one_minus_cos_theta_by_theta * vec_a[1],
        one_minus_cos_theta_by_theta * vec_a[0] +
            sin_theta_by_theta * vec_a[1]);
    return Se2<Scalar>(so2, trans);
  }

  /// Returns closest Se3 given arbitrary 4x4 matrix.
  ///
  template <class TS = Scalar>
  static SOPHUS_FUNC std::enable_if_t<std::is_floating_point<TS>::value, Se2>
  fitToSe2(Eigen::Matrix3<Scalar> const& mat3x3) {
    return Se2(
        So2<Scalar>::fitToSo2(mat3x3.template block<2, 2>(0, 0)),
        mat3x3.template block<2, 1>(0, 2));
  }

  /// Returns the ith infinitesimal generators of SE(2).
  ///
  /// The infinitesimal generators of SE(2) are:
  ///
  /// ```
  ///         |  0  0  1 |
  ///   G_0 = |  0  0  0 |
  ///         |  0  0  0 |
  ///
  ///         |  0  0  0 |
  ///   G_1 = |  0  0  1 |
  ///         |  0  0  0 |
  ///
  ///         |  0 -1  0 |
  ///   G_2 = |  1  0  0 |
  ///         |  0  0  0 |
  /// ```
  ///
  /// Precondition: ``i`` must be in 0, 1 or 2.
  ///
  SOPHUS_FUNC static Transformation generator(int i) {
    FARM_CHECK(i >= 0 || i <= 2, "i should be in range [0,2].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  /// hat-operator
  ///
  /// It takes in the 3-vector representation (= twist) and returns the
  /// corresponding matrix representation of Lie algebra element.
  ///
  /// Formally, the hat()-operator of SE(3) is defined as
  ///
  ///   ``hat(.): R^3 -> R^{3x33},  hat(a) = sum_i a_i * G_i``  (for i=0,1,2)
  ///
  /// with ``G_i`` being the ith infinitesimal generator of SE(2).
  ///
  /// The corresponding inverse is the vee()-operator, see below.
  ///
  SOPHUS_FUNC static Transformation hat(Tangent const& vec_a) {
    Transformation mat_omega;
    mat_omega.setZero();
    mat_omega.template topLeftCorner<2, 2>() = So2<Scalar>::hat(vec_a[2]);
    mat_omega.col(2).template head<2>() = vec_a.template head<2>();
    return mat_omega;
  }

  /// Lie bracket
  ///
  /// It computes the Lie bracket of SE(2). To be more specific, it computes
  ///
  ///   ``[omega_1, omega_2]_se2 := vee([hat(omega_1), hat(omega_2)])``
  ///
  /// with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.)`` the
  /// hat()-operator and ``vee(.)`` the vee()-operator of SE(2).
  ///
  SOPHUS_FUNC static Tangent lieBracket(
      Tangent const& vec_a, Tangent const& vec_b) {
    Eigen::Vector2<Scalar> upsilon1 = vec_a.template head<2>();
    Eigen::Vector2<Scalar> upsilon2 = vec_b.template head<2>();
    Scalar theta1 = vec_a[2];
    Scalar theta2 = vec_b[2];

    return Tangent(
        -theta1 * upsilon2[1] + theta2 * upsilon1[1],
        theta1 * upsilon2[0] - theta2 * upsilon1[0],
        Scalar(0));
  }

  /// Construct pure rotation.
  ///
  static SOPHUS_FUNC Se2 rot(Scalar const& x) {
    return Se2(So2<Scalar>(x), Eigen::Vector2<Scalar>::Zero());
  }

  /// Draw uniform sample from SE(2) manifold.
  ///
  /// Translations are drawn component-wise from the range [-1, 1].
  ///
  template <class TUniformRandomBitGenerator>
  static Se2 sampleUniform(TUniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    return Se2(
        So2<Scalar>::sampleUniform(generator),
        Eigen::Vector2<Scalar>(uniform(generator), uniform(generator)));
  }

  /// Construct a translation only SE(2) instance.
  ///
  template <class TX, class TY>
  static SOPHUS_FUNC Se2 trans(TX const& x, TY const& y) {
    return Se2(So2<Scalar>(), Eigen::Vector2<Scalar>(x, y));
  }

  static SOPHUS_FUNC Se2 trans(Eigen::Vector2<Scalar> const& xy) {
    return Se2(So2<Scalar>(), xy);
  }

  /// Construct x-axis translation.
  ///
  static SOPHUS_FUNC Se2 transX(Scalar const& x) {
    return Se2::trans(x, Scalar(0));
  }

  /// Construct y-axis translation.
  ///
  static SOPHUS_FUNC Se2 transY(Scalar const& y) {
    return Se2::trans(Scalar(0), y);
  }

  /// vee-operator
  ///
  /// It takes the 3x3-matrix representation ``Omega`` and maps it to the
  /// corresponding 3-vector representation of Lie algebra.
  ///
  /// This is the inverse of the hat()-operator, see above.
  ///
  /// Precondition: ``Omega`` must have the following structure:
  ///
  ///                |  0 -d  a |
  ///                |  d  0  b |
  ///                |  0  0  0 |
  ///
  SOPHUS_FUNC static Tangent vee(Transformation const& omega) {
    FARM_CHECK(
        omega.row(2).template lpNorm<1>() < kEpsilon<Scalar>,
        "Omega: \n{}",
        omega);
    Tangent upsilon_omega;
    upsilon_omega.template head<2>() = omega.col(2).template head<2>();
    upsilon_omega[2] = So2<Scalar>::vee(omega.template topLeftCorner<2, 2>());
    return upsilon_omega;
  }

 protected:
  SO2Member so2_;                  // NOLINT
  TranslationMember translation_;  // NOLINT
};

template <class TScalar, int kOptions>
SOPHUS_FUNC Se2<TScalar, kOptions>::Se2()
    : translation_(TranslationMember::Zero()) {
  static_assert(
      std::is_standard_layout<Se2>::value,
      "Assume standard layout for the use of offsetof check below.");
  static_assert(
      offsetof(Se2, so2_) + sizeof(Scalar) * So2<Scalar>::kNumParameters ==
          offsetof(Se2, translation_),
      "This class assumes packed storage and hence will only work "
      "correctly depending on the compiler (options) - in "
      "particular when using [this->data(), this-data() + "
      "kNumParameters] to access the raw data in a contiguous fashion.");
}

}  // namespace sophus

namespace Eigen {  // NOLINT

/// Specialization of Eigen::Map for ``Se2``; derived from Se2Base.
///
/// Allows us to wrap Se2 objects around POD array.
template <class TScalar, int kOptions>
class Map<sophus::Se2<TScalar>, kOptions>
    : public sophus::Se2Base<Map<sophus::Se2<TScalar>, kOptions>> {
 public:
  using Base = sophus::Se2Base<Map<sophus::Se2<TScalar>, kOptions>>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  explicit Map(Scalar* coeffs)
      : so2_(coeffs),
        translation_(coeffs + sophus::So2<Scalar>::kNumParameters) {}

  /// Mutator of So3
  ///
  SOPHUS_FUNC Map<sophus::So2<Scalar>, kOptions>& so2() { return so2_; }

  /// Accessor of So3
  ///
  SOPHUS_FUNC [[nodiscard]] Map<sophus::So2<Scalar>, kOptions> const& so2()
      const {
    return so2_;
  }

  /// Mutator of translation vector
  ///
  SOPHUS_FUNC Map<Eigen::Vector2<Scalar>, kOptions>& translation() {
    return translation_;
  }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector2<Scalar>, kOptions> const&
  translation() const {
    return translation_;
  }

 protected:
  Map<sophus::So2<Scalar>, kOptions> so2_;             // NOLINT
  Map<Eigen::Vector2<Scalar>, kOptions> translation_;  // NOLINT
};

/// Specialization of Eigen::Map for ``Se2 const``; derived from Se2Base.
///
/// Allows us to wrap Se2 objects around POD array.
template <class TScalar, int kOptions>
class Map<sophus::Se2<TScalar> const, kOptions>
    : public sophus::Se2Base<Map<sophus::Se2<TScalar> const, kOptions>> {
 public:
  using Base = sophus::Se2Base<Map<sophus::Se2<TScalar> const, kOptions>>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar const* coeffs)
      : so2_(coeffs),
        translation_(coeffs + sophus::So2<Scalar>::kNumParameters) {}

  /// Accessor of So3
  ///
  SOPHUS_FUNC [[nodiscard]] Map<sophus::So2<Scalar> const, kOptions> const&
  so2() const {
    return so2_;
  }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector2<Scalar> const, kOptions> const&
  translation() const {
    return translation_;
  }

 protected:
  Map<sophus::So2<Scalar> const, kOptions> so2_;             // NOLINT
  Map<Eigen::Vector2<Scalar> const, kOptions> translation_;  // NOLINT
};
}  // namespace Eigen
