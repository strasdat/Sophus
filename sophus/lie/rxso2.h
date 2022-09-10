// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Direct product R X SO(2) - rotation and scaling in 2d.

#pragma once

#include "so2.h"

namespace sophus {
template <class ScalarT, int kOptions = 0>
class RxSo2;
using RxSo2F64 = RxSo2<double>;
using RxSo2F32 = RxSo2<float>;

template <class ScalarT, int kOptions = 0>
/* [[deprecated]] */ using RxSO2 = RxSo2<ScalarT, kOptions>;
/* [[deprecated]] */ using RxSO2d = RxSo2F64;
/* [[deprecated]] */ using RxSO2f = RxSo2F32;
}  // namespace sophus

namespace Eigen {  // NOLINT
namespace internal {

template <class ScalarT, int kOptionsT>
struct traits<sophus::RxSo2<ScalarT, kOptionsT>> {
  static constexpr int kOptions = kOptionsT;
  using Scalar = ScalarT;
  using ComplexType = Eigen::Matrix<Scalar, 2, 1, kOptions>;
};

template <class ScalarT, int kOptionsT>
struct traits<Map<sophus::RxSo2<ScalarT>, kOptionsT>>
    : traits<sophus::RxSo2<ScalarT, kOptionsT>> {
  static constexpr int kOptions = kOptionsT;
  using Scalar = ScalarT;
  using ComplexType = Map<Eigen::Vector2<Scalar>, kOptions>;
};

template <class ScalarT, int kOptionsT>
struct traits<Map<sophus::RxSo2<ScalarT> const, kOptionsT>>
    : traits<sophus::RxSo2<ScalarT, kOptionsT> const> {
  static constexpr int kOptions = kOptionsT;
  using Scalar = ScalarT;
  using ComplexType = Map<Eigen::Vector2<Scalar> const, kOptions>;
};
}  // namespace internal
}  // namespace Eigen

namespace sophus {

/// RxSo2 base type - implements RxSo2 class but is storage agnostic
///
/// This class implements the group ``R+ x SO(2)``, the direct product of the
/// group of positive scalar 2x2 matrices (= isomorph to the positive
/// real numbers) and the two-dimensional special orthogonal group SO(2).
/// Geometrically, it is the group of rotation and scaling in two dimensions.
/// As a matrix groups, R+ x SO(2) consists of matrices of the form ``s * R``
/// where ``R`` is an orthogonal matrix with ``det(R) = 1`` and ``s > 0``
/// being a positive real number. In particular, it has the following form:
///
///     | s * cos(theta)  s * -sin(theta) |
///     | s * sin(theta)  s *  cos(theta) |
///
/// where ``theta`` being the rotation angle. Internally, it is represented by
/// the first column of the rotation matrix, or in other words by a non-zero
/// complex number.
///
/// R+ x SO(2) is not compact, but a commutative group. First it is not compact
/// since the scale factor is not bound. Second it is commutative since
/// ``sR(alpha, s1) * sR(beta, s2) = sR(beta, s2) * sR(alpha, s1)``,  simply
/// because ``alpha + beta = beta + alpha`` and ``s1 * s2 = s2 * s1`` with
/// ``alpha`` and ``beta`` being rotation angles and ``s1``, ``s2`` being scale
/// factors.
///
/// This class has the explicit class invariant that the scale ``s`` is not
/// too close to either zero or infinity. Strictly speaking, it must hold that:
///
///   ``complex().norm() >= Constants::epsilon()`` and
///   ``1. / complex().norm() >= Constants::epsilon()``.
///
/// In order to obey this condition, group multiplication is implemented with
/// saturation such that a product always has a scale which is equal or greater
/// this threshold.
template <class DerivedT>
class RxSo2Base {
 public:
  static constexpr int kOptions = Eigen::internal::traits<DerivedT>::kOptions;
  using Scalar = typename Eigen::internal::traits<DerivedT>::Scalar;
  using ComplexType = typename Eigen::internal::traits<DerivedT>::ComplexType;
  using ComplexTemporaryType = Eigen::Matrix<Scalar, 2, 1, kOptions>;

  /// Degrees of freedom of manifold, number of dimensions in tangent space
  /// (one for rotation and one for scaling).
  static int constexpr kDoF = 2;
  /// Number of internal parameters used (complex number is a tuple).
  static int constexpr kNumParameters = 2;
  /// Group transformations are 2x2 matrices.
  static int constexpr kMatrixDim = 2;
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
  /// double scalars with RxSo2 operations.
  template <typename OtherDerivedT>
  using ReturnScalar = typename Eigen::
      ScalarBinaryOpTraits<Scalar, typename OtherDerivedT::Scalar>::ReturnType;

  template <typename OtherDerivedT>
  using RxSo2Product = RxSo2<ReturnScalar<OtherDerivedT>>;

  template <typename PointDerivedT>
  using PointProduct = Eigen::Vector2<ReturnScalar<PointDerivedT>>;

  template <typename HPointDerivedT>
  using HomogeneousPointProduct = Eigen::Vector3<ReturnScalar<HPointDerivedT>>;

  /// Adjoint transformation
  ///
  /// This function return the adjoint transformation ``Ad`` of the group
  /// element ``A`` such that for all ``x`` it holds that
  /// ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  ///
  /// For RxSO(2), it simply returns the identity matrix.
  ///
  SOPHUS_FUNC [[nodiscard]] Adjoint adj() const { return Adjoint::Identity(); }

  /// Returns rotation angle.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar angle() const {
    return So2<Scalar>(complex()).log();
  }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class NewScalarTypeT>
  SOPHUS_FUNC [[nodiscard]] RxSo2<NewScalarTypeT> cast() const {
    typename RxSo2<NewScalarTypeT>::ComplexType c =
        complex().template cast<NewScalarTypeT>();
    return RxSo2<NewScalarTypeT>(c);
  }

  /// This provides unsafe read/write access to internal data. RxSO(2) is
  /// represented by a complex number (two parameters). When using direct
  /// write access, the user needs to take care of that the complex number is
  /// not set close to zero.
  ///
  /// Note: The first parameter represents the real part, while the
  /// second parameter represent the imaginary part.
  ///
  SOPHUS_FUNC Scalar* data() { return mutComplex().data(); }

  /// Const version of data() above.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar const* data() const {
    return complex().data();
  }

  /// Returns group inverse.
  ///
  SOPHUS_FUNC [[nodiscard]] RxSo2<Scalar> inverse() const {
    Scalar squared_scale = complex().squaredNorm();
    Eigen::Vector2<Scalar> xy = complex() / squared_scale;
    return RxSo2<Scalar>(xy.x(), -xy.y());
  }

  /// Logarithmic map
  ///
  /// Computes the logarithm, the inverse of the group exponential which maps
  /// element of the group (scaled rotation matrices) to elements of the tangent
  /// space (rotation-vector plus logarithm of scale factor).
  ///
  /// To be specific, this function computes ``vee(logmat(.))`` with
  /// ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  /// of RxSo2.
  ///
  SOPHUS_FUNC [[nodiscard]] Tangent log() const {
    using std::log;
    Tangent theta_sigma;
    theta_sigma[1] = log(scale());
    theta_sigma[0] = So2<Scalar>(complex()).log();
    return theta_sigma;
  }

  /// Returns 2x2 matrix representation of the instance.
  ///
  /// For RxSo2, the matrix representation is an scaled orthogonal matrix ``sR``
  /// with ``det(R)=s^2``, thus a scaled rotation matrix ``R``  with scale
  /// ``s``.
  ///
  SOPHUS_FUNC [[nodiscard]] Transformation matrix() const {
    Transformation s_r;
    // clang-format off
    s_r << complex()[0], -complex()[1],
          complex()[1],  complex()[0];
    // clang-format on
    return s_r;
  }

  /// Assignment-like operator from OtherDerived.
  ///
  template <class OtherDerivedT>
  SOPHUS_FUNC RxSo2Base<DerivedT>& operator=(
      RxSo2Base<OtherDerivedT> const& other) {
    mutComplex() = other.complex();
    return *this;
  }

  /// Group multiplication, which is rotation concatenation and scale
  /// multiplication.
  ///
  /// Note: This function performs saturation for products close to zero in
  /// order to ensure the class invariant.
  ///
  template <typename OtherDerivedT>
  SOPHUS_FUNC RxSo2Product<OtherDerivedT> operator*(
      RxSo2Base<OtherDerivedT> const& other) const {
    using ResultT = ReturnScalar<OtherDerivedT>;

    Scalar lhs_real = complex().x();
    Scalar lhs_imag = complex().y();
    typename OtherDerivedT::Scalar const& rhs_real = other.complex().x();
    typename OtherDerivedT::Scalar const& rhs_imag = other.complex().y();
    /// complex multiplication
    typename RxSo2Product<OtherDerivedT>::ComplexType result_complex(
        lhs_real * rhs_real - lhs_imag * rhs_imag,
        lhs_real * rhs_imag + lhs_imag * rhs_real);

    const ResultT squared_scale = result_complex.squaredNorm();

    if (squared_scale < kEpsilon<ResultT> * kEpsilon<ResultT>) {
      /// Saturation to ensure class invariant.
      result_complex.normalize();
      result_complex *= kEpsilonPlus<ResultT>;
    }
    if (squared_scale > Scalar(1.) / (kEpsilon<ResultT> * kEpsilon<ResultT>)) {
      /// Saturation to ensure class invariant.
      result_complex.normalize();
      result_complex /= kEpsilonPlus<ResultT>;
    }
    return RxSo2Product<OtherDerivedT>(result_complex);
  }

  /// Group action on 2-points.
  ///
  /// This function rotates a 2 dimensional point ``p`` by the So2 element
  /// ``bar_R_foo`` (= rotation matrix) and scales it by the scale factor ``s``:
  ///
  ///   ``p_bar = s * (bar_R_foo * p_foo)``.
  ///
  template <
      typename PointDerivedT,
      typename = typename std::enable_if<
          IsFixedSizeVector<PointDerivedT, 2>::value>::type>
  SOPHUS_FUNC PointProduct<PointDerivedT> operator*(
      Eigen::MatrixBase<PointDerivedT> const& p) const {
    return matrix() * p;
  }

  /// Group action on homogeneous 2-points. See above for more details.
  ///
  template <
      typename HPointDerivedT,
      typename = typename std::enable_if<
          IsFixedSizeVector<HPointDerivedT, 3>::value>::type>
  SOPHUS_FUNC HomogeneousPointProduct<HPointDerivedT> operator*(
      Eigen::MatrixBase<HPointDerivedT> const& p) const {
    const auto rsp = *this * p.template head<2>();
    return HomogeneousPointProduct<HPointDerivedT>(rsp(0), rsp(1), p(2));
  }

  /// Group action on lines.
  ///
  /// This function rotates a parameterized line ``l(t) = o + t * d`` by the So2
  /// element and scales it by the scale factor
  ///
  /// Origin ``o`` is rotated and scaled
  /// Direction ``d`` is rotated (preserving it's norm)
  ///
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), (*this) * l.direction() / scale());
  }

  /// Group action on hyper-planes.
  ///
  /// This function rotates a hyper-plane ``n.x + d = 0`` by the So2
  /// element and scales offset by the scale factor
  ///
  /// Normal vector ``n`` is rotated
  /// Offset ``d`` is scaled
  ///
  /// Note that in 2d-case hyper-planes are just another parametrization of
  /// lines
  ///
  SOPHUS_FUNC Hyperplane operator*(Hyperplane const& p) const {
    const auto this_scale = scale();
    return Hyperplane(
        (*this) * p.normal() / this_scale, this_scale * p.offset());
  }

  /// In-place group multiplication. This method is only valid if the return
  /// type of the multiplication is compatible with this So2's Scalar type.
  ///
  /// Note: This function performs saturation for products close to zero in
  /// order to ensure the class invariant.
  ///
  template <
      typename OtherDerivedT,
      typename = typename std::enable_if<
          std::is_same<Scalar, ReturnScalar<OtherDerivedT>>::value>::type>
  SOPHUS_FUNC RxSo2Base<DerivedT>& operator*=(
      RxSo2Base<OtherDerivedT> const& other) {
    *static_cast<DerivedT*>(this) = *this * other;
    return *this;
  }

  /// Returns derivative of  this * RxSo2::exp(x) wrt. x at x=0
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kNumParameters, kDoF>
  dxThisMulExpXAt0() const {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> d;
    // clang-format off
    d << -complex().y(), complex().x(),
            complex().x(), complex().y();
    // clang-format on
    return d;
  }

  /// Returns derivative of log(this^{-1} * x) by x at x=this.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kDoF, kNumParameters>
  dxLogThisInvTimesXAtThis() const {
    Eigen::Matrix<Scalar, kDoF, kNumParameters> d;
    const Scalar norm_sq_inv = Scalar(1.) / complex().squaredNorm();
    // clang-format off
    d << -complex().y(), complex().x(),
            complex().x(), complex().y();
    // clang-format on
    return d * norm_sq_inv;
  }

  /// Returns internal parameters of RxSO(2).
  ///
  /// It returns (c[0], c[1]), with c being the  complex number.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Vector<Scalar, kNumParameters> params()
      const {
    return complex();
  }

  /// Sets non-zero complex
  ///
  /// Precondition: ``z`` must not be close to either zero or infinity.
  SOPHUS_FUNC void setComplex(Eigen::Vector2<Scalar> const& z) {
    FARM_CHECK(
        z.squaredNorm() > kEpsilon<Scalar> * kEpsilon<Scalar>,
        "Scale factor must be greater-equal epsilon.");
    FARM_CHECK(
        z.squaredNorm() < Scalar(1.) / (kEpsilon<Scalar> * kEpsilon<Scalar>),
        "Inverse scale factor must be greate-equal epsilon.");
    static_cast<DerivedT*>(this)->mutComplex() = z;
  }

  /// Accessor of complex.
  ///
  SOPHUS_FUNC [[nodiscard]] ComplexType const& complex() const {
    return static_cast<DerivedT const*>(this)->complex();
  }

  /// Returns rotation matrix.
  ///
  SOPHUS_FUNC [[nodiscard]] Transformation rotationMatrix() const {
    ComplexTemporaryType norm_quad = complex();
    norm_quad.normalize();
    return So2<Scalar>(norm_quad).matrix();
  }

  /// Returns scale.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar scale() const {
    using std::hypot;
    return hypot(complex().x(), complex().y());
  }

  /// Setter of rotation angle, leaves scale as is.
  ///
  SOPHUS_FUNC void setAngle(Scalar const& theta) { setSO2(So2<Scalar>(theta)); }

  /// Setter of complex using rotation matrix ``R``, leaves scale as is.
  ///
  /// Precondition: ``R`` must be orthogonal with determinant of one.
  ///
  SOPHUS_FUNC void setRotationMatrix(Transformation const& mat_r) {
    setSO2(So2<Scalar>(mat_r));
  }

  /// Sets scale and leaves rotation as is.
  ///
  SOPHUS_FUNC void setScale(Scalar const& scale) {
    using std::sqrt;
    mutComplex().normalize();
    mutComplex() *= scale;
  }

  /// Setter of complex number using scaled rotation matrix ``sR``.
  ///
  /// Precondition: The 2x2 matrix must be "scaled orthogonal"
  ///               and have a positive determinant.
  ///
  SOPHUS_FUNC void setScaledRotationMatrix(
      Transformation const& mat_scaled_rot) {
    FARM_CHECK(
        isScaledOrthogonalAndPositive(mat_scaled_rot),
        "mat_scaled_rot must be scaled orthogonal:\n {}",
        mat_scaled_rot);
    mutComplex() = mat_scaled_rot.col(0);
  }

  /// Setter of SO(2) rotations, leaves scale as is.
  ///
  SOPHUS_FUNC void setSO2(So2<Scalar> const& so2) {
    using std::sqrt;
    Scalar saved_scale = scale();
    mutComplex() = so2.unitComplex();
    mutComplex() *= saved_scale;
  }

  SOPHUS_FUNC [[nodiscard]] So2<Scalar> so2() const {
    return So2<Scalar>(complex());
  }

 private:
  /// Mutator of complex is private to ensure class invariant.
  ///
  SOPHUS_FUNC ComplexType& mutComplex() {
    return static_cast<DerivedT*>(this)->mutComplex();
  }
};

/// RxSo2 using storage; derived from RxSo2Base.
template <class ScalarT, int kOptions>
class RxSo2 : public RxSo2Base<RxSo2<ScalarT, kOptions>> {
 public:
  using Base = RxSo2Base<RxSo2<ScalarT, kOptions>>;
  using Scalar = ScalarT;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using ComplexMember = Eigen::Matrix<Scalar, 2, 1, kOptions>;

  /// ``Base`` is friend so complex_nonconst can be accessed from ``Base``.
  friend class RxSo2Base<RxSo2<ScalarT, kOptions>>;

  using Base::operator=;

  /// Define copy-assignment operator explicitly. The definition of
  /// implicit copy assignment operator is deprecated in presence of a
  /// user-declared copy constructor (-Wdeprecated-copy in clang >= 13).
  SOPHUS_FUNC RxSo2& operator=(RxSo2 const& other) = default;

  static int constexpr kDoF = Base::kDoF;
  static int constexpr kNumParameters = Base::kNumParameters;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Default constructor initializes complex number to identity rotation and
  /// scale to 1.
  ///
  SOPHUS_FUNC RxSo2() : complex_(Scalar(1), Scalar(0)) {}

  /// Copy constructor
  ///
  SOPHUS_FUNC RxSo2(RxSo2 const& other) = default;

  /// Copy-like constructor from OtherDerived.
  ///
  template <class OtherDerivedT>
  SOPHUS_FUNC RxSo2(RxSo2Base<OtherDerivedT> const& other)
      : complex_(other.complex()) {}

  /// Constructor from scaled rotation matrix s*R
  ///
  /// Precondition: rotation matrix need to be scaled orthogonal with
  /// determinant of ``s^2``.
  ///
  SOPHUS_FUNC explicit RxSo2(Transformation const& mat_r) {
    this->setScaledRotationMatrix(mat_r);
  }

  /// Constructor from scale factor and rotation matrix ``R``.
  ///
  /// Precondition: Rotation matrix ``R`` must to be orthogonal with determinant
  ///               of 1 and ``scale`` must not to be close to either zero or
  ///               infinity.
  ///
  SOPHUS_FUNC RxSo2(Scalar const& scale, Transformation const& mat_r)
      : RxSo2((scale * So2<Scalar>(mat_r).unitComplex()).eval()) {}

  /// Constructor from scale factor and So2
  ///
  /// Precondition: ``scale`` must not be close to either zero or infinity.
  ///
  SOPHUS_FUNC RxSo2(Scalar const& scale, So2<Scalar> const& so2)
      : RxSo2((scale * so2.unitComplex()).eval()) {}

  /// Constructor from complex number.
  ///
  /// Precondition: complex number must not be close to either zero or infinity
  ///
  SOPHUS_FUNC explicit RxSo2(Eigen::Vector2<Scalar> const& z) : complex_(z) {
    FARM_CHECK(
        complex_.squaredNorm() >= kEpsilon<Scalar> * kEpsilon<Scalar>,
        "Scale factor must be greater-equal epsilon: {} vs {}",
        complex_.squaredNorm(),
        kEpsilon<Scalar> * kEpsilon<Scalar>);
    FARM_CHECK(
        complex_.squaredNorm() <=
            Scalar(1.) / (kEpsilon<Scalar> * kEpsilon<Scalar>),
        "Inverse scale factor must be greater-equal epsilon: % vs %",
        Scalar(1.) / complex_.squaredNorm(),
        kEpsilon<Scalar> * kEpsilon<Scalar>);
  }

  /// Constructor from complex number.
  ///
  /// Precondition: complex number must not be close to either zero or inifnity.
  ///
  SOPHUS_FUNC explicit RxSo2(Scalar const& real, Scalar const& imag)
      : RxSo2(Eigen::Vector2<Scalar>(real, imag)) {}

  /// Accessor of complex.
  ///
  SOPHUS_FUNC [[nodiscard]] ComplexMember const& complex() const {
    return complex_;
  }

  /// Returns derivative of exp(x) wrt. ``x``
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpX(
      Tangent const& a) {
    using std::cos;
    using std::exp;
    using std::sin;
    Scalar const theta = a[0];
    Scalar const sigma = a[1];

    Eigen::Matrix<Scalar, kNumParameters, kDoF> d;
    // clang-format off
    d << -sin(theta), cos(theta),
            cos(theta), sin(theta);
    // clang-format on
    return d * exp(sigma);
  }

  /// Returns derivative of exp(x) wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpXAt0() {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> d;
    Scalar const i(1.);
    Scalar const o(0.);
    // clang-format off
    d << o, i,
         i, o;
    // clang-format on
    return d;
  }

  /// Returns derivative of exp(x) * p wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, 2, kDoF> dxExpXTimesPointAt0(
      Point const& point) {
    Eigen::Matrix<Scalar, 2, kDoF> d;
    d << sophus::So2<Scalar>::dxExpXTimesPointAt0(point), point;
    return d;
  }

  /// Returns derivative of ``expmat(x)`` wrt. ``x_i at x=0``, with
  /// ``expmat(.)`` being the matrix exponential.
  ///
  SOPHUS_FUNC static Transformation dxiExpmatXAt0(int i) {
    return generator(i);
  }

  /// Group exponential
  ///
  /// This functions takes in an element of tangent space (= rotation angle
  /// plus logarithm of scale) and returns the corresponding element of the
  /// group RxSo2.
  ///
  /// To be more specific, this function computes ``expmat(hat(theta))``
  /// with ``expmat(.)`` being the matrix exponential and ``hat(.)`` being the
  /// hat()-operator of RSO2.
  ///
  SOPHUS_FUNC static RxSo2<Scalar> exp(Tangent const& vec_a) {
    using std::exp;
    using std::max;
    using std::min;

    Scalar const theta = vec_a[0];
    Scalar const sigma = vec_a[1];
    Scalar s = exp(sigma);
    // Ensuring proper scale
    s = max(s, kEpsilonPlus<Scalar>);
    s = min(s, Scalar(1.) / kEpsilonPlus<Scalar>);
    Eigen::Vector2<Scalar> z = So2<Scalar>::exp(theta).unitComplex();
    z *= s;
    return RxSo2<Scalar>(z);
  }

  /// Returns the ith infinitesimal generators of ``R+ x SO(2)``.
  ///
  /// The infinitesimal generators of RxSo2 are:
  ///
  /// ```
  ///         |  0 -1 |
  ///   G_0 = |  1  0 |
  ///
  ///         |  1  0 |
  ///   G_1 = |  0  1 |
  /// ```
  ///
  /// Precondition: ``i`` must be 0, or 1.
  ///
  SOPHUS_FUNC static Transformation generator(int i) {
    FARM_CHECK(i >= 0 && i <= 1, "i should be 0 or 1.");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  /// hat-operator
  ///
  /// It takes in the 2-vector representation ``a`` (= rotation angle plus
  /// logarithm of scale) and  returns the corresponding matrix representation
  /// of Lie algebra element.
  ///
  /// Formally, the hat()-operator of RxSo2 is defined as
  ///
  ///   ``hat(.): R^2 -> R^{2x2},  hat(a) = sum_i a_i * G_i``  (for i=0,1,2)
  ///
  /// with ``G_i`` being the ith infinitesimal generator of RxSo2.
  ///
  /// The corresponding inverse is the vee()-operator, see below.
  ///
  SOPHUS_FUNC static Transformation hat(Tangent const& vec_a) {
    Transformation mat_a;
    // clang-format off
    mat_a << vec_a(1), -vec_a(0),
             vec_a(0),  vec_a(1);
    // clang-format on
    return mat_a;
  }

  /// Lie bracket
  ///
  /// It computes the Lie bracket of RxSO(2). To be more specific, it computes
  ///
  ///   ``[omega_1, omega_2]_rxso2 := vee([hat(omega_1), hat(omega_2)])``
  ///
  /// with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.)`` the
  /// hat()-operator and ``vee(.)`` the vee()-operator of RxSo2.
  ///
  SOPHUS_FUNC static Tangent lieBracket(
      Tangent const& /*unused*/, Tangent const& /*unused*/) {
    Eigen::Vector2<Scalar> res;
    res.setZero();
    return res;
  }

  /// Draw uniform sample from RxSO(2) manifold.
  ///
  /// The scale factor is drawn uniformly in log2-space from [-1, 1],
  /// hence the scale is in [0.5, 2)].
  ///
  template <class UniformRandomBitGeneratorT>
  static RxSo2 sampleUniform(UniformRandomBitGeneratorT& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    using std::exp2;
    return RxSo2(
        exp2(uniform(generator)), So2<Scalar>::sampleUniform(generator));
  }

  /// vee-operator
  ///
  /// It takes the 2x2-matrix representation ``Omega`` and maps it to the
  /// corresponding vector representation of Lie algebra.
  ///
  /// This is the inverse of the hat()-operator, see above.
  ///
  /// Precondition: ``Omega`` must have the following structure:
  ///
  ///                |  d -x |
  ///                |  x  d |
  ///
  SOPHUS_FUNC static Tangent vee(Transformation const& omega) {
    using std::abs;
    return Tangent(omega(1, 0), omega(0, 0));
  }

 protected:
  SOPHUS_FUNC ComplexMember& mutComplex() { return complex_; }

  ComplexMember complex_;  // NOLINT
};

}  // namespace sophus

namespace Eigen {  // NOLINT

/// Specialization of Eigen::Map for ``RxSo2``; derived from  RxSo2Base.
///
/// Allows us to wrap RxSo2 objects around POD array (e.g. external z style
/// complex).
template <class ScalarT, int kOptions>
class Map<sophus::RxSo2<ScalarT>, kOptions>
    : public sophus::RxSo2Base<Map<sophus::RxSo2<ScalarT>, kOptions>> {
  using Base = sophus::RxSo2Base<Map<sophus::RxSo2<ScalarT>, kOptions>>;

 public:
  using Scalar = ScalarT;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  /// ``Base`` is friend so complex_nonconst can be accessed from ``Base``.
  friend class sophus::RxSo2Base<Map<sophus::RxSo2<ScalarT>, kOptions>>;

  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar* coeffs) : complex_(coeffs) {}

  /// Accessor of complex.
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector2<Scalar>, kOptions> const&
  complex() const {
    return complex_;
  }

 protected:
  SOPHUS_FUNC Map<Eigen::Vector2<Scalar>, kOptions>& mutComplex() {
    return complex_;
  }

  Map<Eigen::Vector2<Scalar>, kOptions> complex_;  // NOLINT
};

/// Specialization of Eigen::Map for ``RxSo2 const``; derived from  RxSo2Base.
///
/// Allows us to wrap RxSo2 objects around POD array (e.g. external z style
/// complex).
template <class ScalarT, int kOptions>
class Map<sophus::RxSo2<ScalarT> const, kOptions>
    : public sophus::RxSo2Base<Map<sophus::RxSo2<ScalarT> const, kOptions>> {
 public:
  using Base = sophus::RxSo2Base<Map<sophus::RxSo2<ScalarT> const, kOptions>>;
  using Scalar = ScalarT;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  explicit Map(Scalar const* coeffs) : complex_(coeffs) {}

  /// Accessor of complex.
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector2<Scalar> const, kOptions> const&
  complex() const {
    return complex_;
  }

 protected:
  Map<Eigen::Vector2<Scalar> const, kOptions> complex_;  // NOLINT
};
}  // namespace Eigen
