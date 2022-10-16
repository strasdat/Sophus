// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Similarity group Sim(2) - scaling, rotation and translation in 2d.

#pragma once

#include "sophus/lie/details/sim_impl.h"
#include "sophus/lie/rxso2.h"

namespace sophus {
template <class TScalar, int kOptions = 0>
class Sim2;
using Sim2F64 = Sim2<double>;
using Sim2F32 = Sim2<float>;

/* [[deprecated]] */ using Sim2d = Sim2F64;
/* [[deprecated]] */ using Sim2f = Sim2F32;
}  // namespace sophus

namespace Eigen {  // NOLINT
namespace internal {

template <class TScalar, int kOptions>
struct traits<sophus::Sim2<TScalar, kOptions>> {
  using Scalar = TScalar;
  using TranslationType = Eigen::Matrix<Scalar, 2, 1, kOptions>;
  using RxSo2Type = sophus::RxSo2<Scalar, kOptions>;
};

template <class TScalar, int kOptions>
struct traits<Map<sophus::Sim2<TScalar>, kOptions>>
    : traits<sophus::Sim2<TScalar, kOptions>> {
  using Scalar = TScalar;
  using TranslationType = Map<Eigen::Vector2<Scalar>, kOptions>;
  using RxSo2Type = Map<sophus::RxSo2<Scalar>, kOptions>;
};

template <class TScalar, int kOptions>
struct traits<Map<sophus::Sim2<TScalar> const, kOptions>>
    : traits<sophus::Sim2<TScalar, kOptions> const> {
  using Scalar = TScalar;
  using TranslationType = Map<Eigen::Vector2<Scalar> const, kOptions>;
  using RxSo2Type = Map<sophus::RxSo2<Scalar> const, kOptions>;
};
}  // namespace internal
}  // namespace Eigen

namespace sophus {

/// Sim2 base type - implements Sim2 class but is storage agnostic.
///
/// Sim(2) is the group of rotations  and translation and scaling in 2d. It is
/// the semi-direct product of R+xSO(2) and the 2d Euclidean vector space. The
/// class is represented using a composition of RxSo2  for scaling plus
/// rotation and a 2-vector for translation.
///
///  - 3x3 Eigen::Matrix representation:
///
/// ```
///   | s*R t |
///   |  o  1 |
/// ```
///
/// where ``R`` is a 3x3 rotation matrix, ``s`` a positive scale factor,
/// ``t`` a translation 3-vector and ``o`` a 3-column vector of zeros.
///
///  - Tangent 4-vector: [upsilon, omega, sigma],
///
/// where ``upsilon`` is the translational velocity 3-vector and ``omega`` the
/// rotational velocity 3-vector, and ``sigma = log(s)``.
///
///  - Internal 4-vector representation: [t0, t1, zr, zi],
///
/// with ``t0, t1`` are the translational components, and ``zr``
/// the real part and ``zi`` the imaginary part of a non-zero
/// complex number. Here the scale ``s`` is equal to the norm of the complex
/// number ``s = |z|``.
///
/// See RxSo2 for more details of the scaling + rotation representation in 2d.
///
///
/// Sim(2) is neither compact, nor a commutative group.
///
/// See RxSo2 for more details of the scaling + rotation representation in 2d.
///
template <class TDerived>
class Sim2Base {
 public:
  using Scalar = typename Eigen::internal::traits<TDerived>::Scalar;
  using TranslationType =
      typename Eigen::internal::traits<TDerived>::TranslationType;
  using RxSo2Type = typename Eigen::internal::traits<TDerived>::RxSo2Type;

  /// Degrees of freedom of manifold, number of dimensions in tangent space
  /// (two for translation, one for rotation and one for scaling).
  static int constexpr kDoF = 4;
  /// Number of internal parameters used (2-tuple for complex number, two for
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
  /// double scalars with SIM2 operations.
  template <class TOtherDerived>
  using ReturnScalar = typename Eigen::
      ScalarBinaryOpTraits<Scalar, typename TOtherDerived::Scalar>::ReturnType;

  template <class TOtherDerived>
  using Sim2Product = Sim2<ReturnScalar<TOtherDerived>>;

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
    Adjoint res;
    res.setZero();
    res.template block<2, 2>(0, 0) = rxso2().matrix();
    res(0, 2) = translation()[1];
    res(1, 2) = -translation()[0];
    res.template block<2, 1>(0, 3) = -translation();

    res(2, 2) = Scalar(1);

    res(3, 3) = Scalar(1);
    return res;
  }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class TNewScalarType>
  SOPHUS_FUNC [[nodiscard]] Sim2<TNewScalarType> cast() const {
    return Sim2<TNewScalarType>(
        rxso2().template cast<TNewScalarType>(),
        translation().template cast<TNewScalarType>());
  }

  /// Returns group inverse.
  ///
  SOPHUS_FUNC [[nodiscard]] Sim2<Scalar> inverse() const {
    RxSo2<Scalar> inv_r = rxso2().inverse();
    return Sim2<Scalar>(inv_r, inv_r * (translation() * Scalar(-1)));
  }

  /// Logarithmic map
  ///
  /// Computes the logarithm, the inverse of the group exponential which maps
  /// element of the group (rigid body transformations) to elements of the
  /// tangent space (twist).
  ///
  /// To be specific, this function computes ``vee(logmat(.))`` with
  /// ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  /// of Sim(2).
  ///
  SOPHUS_FUNC [[nodiscard]] Tangent log() const {
    /// The derivation of the closed-form Sim(2) logarithm for is done
    /// analogously to the closed-form solution of the SE(2) logarithm, see
    /// J. Gallier, D. Xu, "Computing exponentials of skew symmetric matrices
    /// and logarithms of orthogonal matrices", IJRA 2002.
    /// https:///pdfs.semanticscholar.org/cfe3/e4b39de63c8cabd89bf3feff7f5449fc981d.pdf
    /// (Sec. 6., pp. 8)
    Tangent res;
    Eigen::Vector2<Scalar> const theta_sigma = rxso2().log();
    Scalar const theta = theta_sigma[0];
    Scalar const sigma = theta_sigma[1];
    Eigen::Matrix2<Scalar> const omega = So2<Scalar>::hat(theta);
    Eigen::Matrix2<Scalar> const w_inv =
        details::calcWInv<Scalar, 2>(omega, theta, sigma, scale());

    res.segment(0, 2) = w_inv * translation();
    res[2] = theta;
    res[3] = sigma;
    return res;
  }

  /// Returns 3x3 matrix representation of the instance.
  ///
  /// It has the following form:
  ///
  ///   | s*R t |
  ///   |  o  1 |
  ///
  /// where ``R`` is a 2x2 rotation matrix, ``s`` a scale factor, ``t`` a
  /// translation 2-vector and ``o`` a 2-column vector of zeros.
  ///
  SOPHUS_FUNC [[nodiscard]] Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.template topLeftCorner<2, 3>() = matrix2x3();
    homogenious_matrix.row(2) =
        Eigen::Matrix<Scalar, 3, 1>(Scalar(0), Scalar(0), Scalar(1));
    return homogenious_matrix;
  }

  /// Returns the significant first two rows of the matrix above.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, 2, 3> matrix2x3() const {
    Eigen::Matrix<Scalar, 2, 3> matrix;
    matrix.template topLeftCorner<2, 2>() = rxso2().matrix();
    matrix.col(2) = translation();
    return matrix;
  }

  /// Assignment-like operator from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Sim2Base<TDerived>& operator=(
      Sim2Base<TOtherDerived> const& other) {
    rxso2() = other.rxso2();
    translation() = other.translation();
    return *this;
  }

  /// Group multiplication, which is rotation plus scaling concatenation.
  ///
  /// Note: That scaling is calculated with saturation. See RxSo2 for
  /// details.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Sim2Product<TOtherDerived> operator*(
      Sim2Base<TOtherDerived> const& other) const {
    return Sim2Product<TOtherDerived>(
        rxso2() * other.rxso2(), translation() + rxso2() * other.translation());
  }

  /// Group action on 2-points.
  ///
  /// This function rotates, scales and translates a two dimensional point
  /// ``p`` by the Sim(2) element ``(bar_sR_foo, t_bar)`` (= similarity
  /// transformation):
  ///
  ///   ``p_bar = bar_sR_foo * p_foo + t_bar``.
  ///
  template <
      typename TPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<TPointDerived, 2>::value>::type>
  SOPHUS_FUNC PointProduct<TPointDerived> operator*(
      Eigen::MatrixBase<TPointDerived> const& p) const {
    return rxso2() * p + translation();
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
        rxso2() * p.template head<2>() + p(2) * translation();
    return HomogeneousPointProduct<THPointDerived>(tp(0), tp(1), p(2));
  }

  /// Group action on lines.
  ///
  /// This function rotates, scales and translates a parametrized line
  /// ``l(t) = o + t * d`` by the Sim(2) element:
  ///
  /// Origin ``o`` is rotated, scaled and translated
  /// Direction ``d`` is rotated
  ///
  SOPHUS_FUNC Line operator*(Line const& l) const {
    Line rotated_line = rxso2() * l;
    return Line(
        rotated_line.origin() + translation(), rotated_line.direction());
  }

  /// Group action on hyper-planes.
  ///
  /// This function rotates a hyper-plane ``n.x + d = 0`` by the Sim2
  /// element:
  ///
  /// Normal vector ``n`` is rotated
  /// Offset ``d`` is scaled and adjusted for translation
  ///
  /// Note that in 2d-case hyper-planes are just another parametrization of
  /// lines
  ///
  SOPHUS_FUNC Hyperplane operator*(Hyperplane const& p) const {
    Hyperplane const rotated = rxso2() * p;
    return Hyperplane(
        rotated.normal(),
        rotated.offset() - translation().dot(rotated.normal()));
  }

  /// Returns internal parameters of Sim(2).
  ///
  /// It returns (c[0], c[1], t[0], t[1]),
  /// with c being the complex number, t the translation 3-vector.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Vector<Scalar, kNumParameters> params()
      const {
    Eigen::Vector<Scalar, kNumParameters> p;
    p << rxso2().params(), translation();
    return p;
  }

  /// In-place group multiplication. This method is only valid if the return
  /// type of the multiplication is compatible with this So2's Scalar type.
  ///
  template <
      typename TOtherDerived,
      typename = typename std::enable_if<
          std::is_same<Scalar, ReturnScalar<TOtherDerived>>::value>::type>
  SOPHUS_FUNC Sim2Base<TDerived>& operator*=(
      Sim2Base<TOtherDerived> const& other) {
    *static_cast<TDerived*>(this) = *this * other;
    return *this;
  }

  /// Returns derivative of  this * Sim2::exp(x)  wrt. x at x=0.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kNumParameters, kDoF>
  dxThisMulExpXAt0() const {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> d;
    d.template block<2, 2>(0, 0).setZero();
    d.template block<2, 2>(0, 2) = rxso2().dxThisMulExpXAt0();
    d.template block<2, 2>(2, 2).setZero();
    d.template block<2, 2>(2, 0) = rxso2().matrix();
    return d;
  }

  /// Returns derivative of log(this^{-1} * x) by x at x=this.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kDoF, kNumParameters>
  dxLogThisInvTimesXAtThis() const {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> d;
    d.template block<2, 2>(0, 0).setZero();
    d.template block<2, 2>(0, 2) = rxso2().inverse().matrix();
    d.template block<2, 2>(2, 0) = rxso2().dxLogThisInvTimesXAtThis();
    d.template block<2, 2>(2, 2).setZero();
    return d;
  }

  /// Setter of non-zero complex number.
  ///
  /// Precondition: ``z`` must not be close to zero.
  ///
  SOPHUS_FUNC void setComplex(Eigen::Vector2<Scalar> const& z) {
    rxso2().setComplex(z);
  }

  /// Accessor of complex number.
  ///
  SOPHUS_FUNC [[nodiscard]]
  typename Eigen::internal::traits<TDerived>::RxSo2Type::ComplexType const&
  complex() const {
    return rxso2().complex();
  }

  /// Returns Rotation matrix
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix2<Scalar> rotationMatrix() const {
    return rxso2().rotationMatrix();
  }

  /// Mutator of So2 group.
  ///
  SOPHUS_FUNC RxSo2Type& rxso2() {
    return static_cast<TDerived*>(this)->rxso2();
  }

  /// Accessor of So2 group.
  ///
  SOPHUS_FUNC [[nodiscard]] RxSo2Type const& rxso2() const {
    return static_cast<TDerived const*>(this)->rxso2();
  }

  /// Returns scale.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar scale() const { return rxso2().scale(); }

  /// Setter of complex number using rotation matrix ``R``, leaves scale as is.
  ///
  SOPHUS_FUNC void setRotationMatrix(Eigen::Matrix2<Scalar>& mat_r) {
    rxso2().setRotationMatrix(mat_r);
  }

  /// Sets scale and leaves rotation as is.
  ///
  /// Note: This function as a significant computational cost, since it has to
  /// call the square root twice.
  ///
  SOPHUS_FUNC void setScale(Scalar const& scale) { rxso2().setScale(scale); }

  /// Setter of complexnumber using scaled rotation matrix ``sR``.
  ///
  /// Precondition: The 2x2 matrix must be "scaled orthogonal"
  ///               and have a positive determinant.
  ///
  SOPHUS_FUNC void setScaledRotationMatrix(
      Eigen::Matrix2<Scalar> const& mat_scaled_rot) {
    rxso2().setScaledRotationMatrix(mat_scaled_rot);
  }

  /// Mutator of translation vector
  ///
  SOPHUS_FUNC TranslationType& translation() {
    return static_cast<TDerived*>(this)->translation();
  }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] TranslationType const& translation() const {
    return static_cast<TDerived const*>(this)->translation();
  }
};

/// Sim2 using default storage; derived from Sim2Base.
template <class TScalar, int kOptions>
class Sim2 : public Sim2Base<Sim2<TScalar, kOptions>> {
 public:
  using Base = Sim2Base<Sim2<TScalar, kOptions>>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using RxSo2Member = RxSo2<Scalar, kOptions>;
  using TranslationMember = Eigen::Matrix<Scalar, 2, 1, kOptions>;

  using Base::operator=;

  /// Define copy-assignment operator explicitly. The definition of
  /// implicit copy assignment operator is deprecated in presence of a
  /// user-declared copy constructor (-Wdeprecated-copy in clang >= 13).
  SOPHUS_FUNC Sim2& operator=(Sim2 const& other) = default;

  static int constexpr kDoF = Base::kDoF;
  static int constexpr kNumParameters = Base::kNumParameters;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Default constructor initializes similarity transform to the identity.
  ///
  SOPHUS_FUNC Sim2();

  /// Copy constructor
  ///
  SOPHUS_FUNC Sim2(Sim2 const& other) = default;

  /// Copy-like constructor from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Sim2(Sim2Base<TOtherDerived> const& other)
      : rxso2_(other.rxso2()), translation_(other.translation()) {
    static_assert(
        std::is_same<typename TOtherDerived::Scalar, Scalar>::value,
        "must be same Scalar type");
  }

  /// Constructor from RxSo2 and translation vector
  ///
  template <class TOtherDerived, class TD>
  SOPHUS_FUNC Sim2(
      RxSo2Base<TOtherDerived> const& rxso2,
      Eigen::MatrixBase<TD> const& translation)
      : rxso2_(rxso2), translation_(translation) {
    static_assert(
        std::is_same<typename TOtherDerived::Scalar, Scalar>::value,
        "must be same Scalar type");
    static_assert(
        std::is_same<typename TD::Scalar, Scalar>::value,
        "must be same Scalar type");
  }

  /// Constructor from complex number and translation vector.
  ///
  /// Precondition: complex number must not be close to zero.
  ///
  template <class TD>
  SOPHUS_FUNC Sim2(
      Eigen::Vector2<Scalar> const& complex_number,
      Eigen::MatrixBase<TD> const& translation)
      : rxso2_(complex_number), translation_(translation) {
    static_assert(
        std::is_same<typename TD::Scalar, Scalar>::value,
        "must be same Scalar type");
  }

  /// Constructor from 3x3 matrix
  ///
  /// Precondition: Top-left 2x2 matrix needs to be "scaled-orthogonal" with
  ///               positive determinant. The last row must be ``(0, 0, 1)``.
  ///
  SOPHUS_FUNC explicit Sim2(Eigen::Matrix<Scalar, 3, 3> const& t)
      : rxso2_((t.template topLeftCorner<2, 2>()).eval()),
        translation_(t.template block<2, 1>(0, 2)) {}

  /// This provides unsafe read/write access to internal data. Sim(2) is
  /// represented by a complex number (two parameters) and a 2-vector. When
  /// using direct write access, the user needs to take care of that the
  /// complex number is not set close to zero.
  ///
  SOPHUS_FUNC Scalar* data() {
    // rxso2_ and translation_ are laid out sequentially with no padding
    return rxso2_.data();
  }

  /// Const version of data() above.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar const* data() const {
    // rxso2_ and translation_ are laid out sequentially with no padding
    return rxso2_.data();
  }

  /// Accessor of RxSo2
  ///
  SOPHUS_FUNC RxSo2Member& rxso2() { return rxso2_; }

  /// Mutator of RxSo2
  ///
  SOPHUS_FUNC [[nodiscard]] RxSo2Member const& rxso2() const { return rxso2_; }

  /// Mutator of translation vector
  ///
  SOPHUS_FUNC TranslationMember& translation() { return translation_; }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] TranslationMember const& translation() const {
    return translation_;
  }

  /// Returns derivative of exp(x) wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpXAt0() {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> j;
    j.template block<2, 2>(0, 0).setZero();
    j.template block<2, 2>(0, 2) = RxSo2<Scalar>::dxExpXAt0();
    j.template block<2, 2>(2, 0).setIdentity();
    j.template block<2, 2>(2, 2).setZero();
    return j;
  }

  /// Returns derivative of exp(x) wrt. x.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpX(
      Tangent const& vec_a) {
    static Eigen::Matrix2<Scalar> const kI = Eigen::Matrix2<Scalar>::Identity();
    static Scalar const kOne(1.0);

    Scalar const theta = vec_a[2];
    Scalar const sigma = vec_a[3];

    Eigen::Matrix2<Scalar> const omega = So2<Scalar>::hat(theta);
    Eigen::Matrix2<Scalar> const omega_dtheta = So2<Scalar>::hat(kOne);
    Eigen::Matrix2<Scalar> const omega2 = omega * omega;
    Eigen::Matrix2<Scalar> const omega2_dtheta =
        omega_dtheta * omega + omega * omega_dtheta;
    Eigen::Matrix2<Scalar> const w =
        details::calcW<Scalar, 2>(omega, theta, sigma);
    Eigen::Vector2<Scalar> const upsilon = vec_a.segment(0, 2);

    Eigen::Matrix<Scalar, kNumParameters, kDoF> j;
    j.template block<2, 2>(0, 0).setZero();
    j.template block<2, 2>(0, 2) =
        RxSo2<Scalar>::dxExpX(vec_a.template tail<2>());
    j.template block<2, 2>(2, 0) = w;

    Scalar a;
    Scalar b;
    Scalar c;
    Scalar a_dtheta;
    Scalar b_dtheta;
    Scalar a_dsigma;
    Scalar b_dsigma;
    Scalar c_dsigma;
    details::calcWDerivatives(
        theta,
        sigma,
        a,
        b,
        c,
        a_dsigma,
        b_dsigma,
        c_dsigma,
        a_dtheta,
        b_dtheta);

    j.template block<2, 1>(2, 2) = (a_dtheta * omega + a * omega_dtheta +
                                    b_dtheta * omega2 + b * omega2_dtheta) *
                                   upsilon;
    j.template block<2, 1>(2, 3) =
        (a_dsigma * omega + b_dsigma * omega2 + c_dsigma * kI) * upsilon;

    return j;
  }

  /// Returns derivative of exp(x) * p wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, 2, kDoF> dxExpXTimesPointAt0(
      Point const& point) {
    Eigen::Matrix<Scalar, 2, kDoF> j;
    j << Eigen::Matrix2<Scalar>::Identity(),
        sophus::RxSo2<Scalar>::dxExpXTimesPointAt0(point);
    return j;
  }

  /// Returns derivative of ``expmat(x)`` wrt. ``x_i at x=0``, with
  /// ``expmat(.)`` being the matrix exponential.
  ///
  SOPHUS_FUNC static Transformation dxiExpmatXAt0(int i) {
    return generator(i);
  }

  /// Derivative of Lie bracket with respect to first element.
  ///
  /// This function returns ``D_a [a, b]`` with ``D_a`` being the
  /// differential operator with respect to ``a``, ``[a, b]`` being the lie
  /// bracket of the Lie algebra sim(2).
  /// See ``lieBracket()`` below.
  ///

  /// Group exponential
  ///
  /// This functions takes in an element of tangent space and returns the
  /// corresponding element of the group Sim(2).
  ///
  /// The first two components of ``a`` represent the translational part
  /// ``upsilon`` in the tangent space of Sim(2), the following two components
  /// of ``a`` represents the rotation ``theta`` and the final component
  /// represents the logarithm of the scaling factor ``sigma``.
  /// To be more specific, this function computes ``expmat(hat(a))`` with
  /// ``expmat(.)`` being the matrix exponential and ``hat(.)`` the hat-operator
  /// of Sim(2), see below.
  ///
  SOPHUS_FUNC static Sim2<Scalar> exp(Tangent const& vec_a) {
    // For the derivation of the exponential map of Sim(kMatrixDim) see
    // H. Strasdat, "Local Accuracy and Global Consistency for Efficient Visual
    // SLAM", PhD thesis, 2012.
    // http:///hauke.strasdat.net/files/strasdat_thesis_2012.pdf (A.5, pp. 186)
    Eigen::Vector2<Scalar> const upsilon = vec_a.segment(0, 2);
    Scalar const theta = vec_a[2];
    Scalar const sigma = vec_a[3];
    RxSo2<Scalar> rxso2 = RxSo2<Scalar>::exp(vec_a.template tail<2>());
    Eigen::Matrix2<Scalar> const omega = So2<Scalar>::hat(theta);
    Eigen::Matrix2<Scalar> const w =
        details::calcW<Scalar, 2>(omega, theta, sigma);
    return Sim2<Scalar>(rxso2, w * upsilon);
  }

  /// Returns the ith infinitesimal generators of Sim(2).
  ///
  /// The infinitesimal generators of Sim(2) are:
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
  ///
  ///         |  1  0  0 |
  ///   G_3 = |  0  1  0 |
  ///         |  0  0  0 |
  /// ```
  ///
  /// Precondition: ``i`` must be in [0, 3].
  ///
  SOPHUS_FUNC static Transformation generator(int i) {
    FARM_CHECK(i >= 0 || i <= 3, "i should be in range [0,3].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  /// hat-operator
  ///
  /// It takes in the 4-vector representation and returns the corresponding
  /// matrix representation of Lie algebra element.
  ///
  /// Formally, the hat()-operator of Sim(2) is defined as
  ///
  ///   ``hat(.): R^4 -> R^{3x3},  hat(a) = sum_i a_i * G_i``  (for i=0,...,6)
  ///
  /// with ``G_i`` being the ith infinitesimal generator of Sim(2).
  ///
  /// The corresponding inverse is the vee()-operator, see below.
  ///
  SOPHUS_FUNC static Transformation hat(Tangent const& vec_a) {
    Transformation omega;
    omega.template topLeftCorner<2, 2>() =
        RxSo2<Scalar>::hat(vec_a.template tail<2>());
    omega.col(2).template head<2>() = vec_a.template head<2>();
    omega.row(2).setZero();
    return omega;
  }

  /// Lie bracket
  ///
  /// It computes the Lie bracket of Sim(2). To be more specific, it computes
  ///
  ///   ``[omega_1, omega_2]_sim2 := vee([hat(omega_1), hat(omega_2)])``
  ///
  /// with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.)`` the
  /// hat()-operator and ``vee(.)`` the vee()-operator of Sim(2).
  ///
  SOPHUS_FUNC static Tangent lieBracket(
      Tangent const& vec_a, Tangent const& vec_b) {
    Eigen::Vector2<Scalar> const vec_upsilon1 = vec_a.template head<2>();
    Eigen::Vector2<Scalar> const vec_upsilon2 = vec_b.template head<2>();
    Scalar const theta1 = vec_a[2];
    Scalar const theta2 = vec_b[2];
    Scalar const sigma1 = vec_a[3];
    Scalar const sigma2 = vec_b[3];

    Tangent res;
    res[0] = -theta1 * vec_upsilon2[1] + theta2 * vec_upsilon1[1] +
             sigma1 * vec_upsilon2[0] - sigma2 * vec_upsilon1[0];
    res[1] = theta1 * vec_upsilon2[0] - theta2 * vec_upsilon1[0] +
             sigma1 * vec_upsilon2[1] - sigma2 * vec_upsilon1[1];
    res[2] = Scalar(0);
    res[3] = Scalar(0);

    return res;
  }

  /// Draw uniform sample from Sim(2) manifold.
  ///
  /// Translations are drawn component-wise from the range [-1, 1].
  /// The scale factor is drawn uniformly in log2-space from [-1, 1],
  /// hence the scale is in [0.5, 2].
  ///
  template <std::uniform_random_bit_generator TUniformRandomBitGenerator>
  static Sim2 sampleUniform(TUniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    return Sim2(
        RxSo2<Scalar>::sampleUniform(generator),
        Eigen::Vector2<Scalar>(uniform(generator), uniform(generator)));
  }

  /// vee-operator
  ///
  /// It takes the 3x3-matrix representation ``Omega`` and maps it to the
  /// corresponding 4-vector representation of Lie algebra.
  ///
  /// This is the inverse of the hat()-operator, see above.
  ///
  /// Precondition: ``Omega`` must have the following structure:
  ///
  ///                |  d -c  a |
  ///                |  c  d  b |
  ///                |  0  0  0 |
  ///
  SOPHUS_FUNC static Tangent vee(Transformation const& mat_omega) {
    Tangent upsilon_omega_sigma;
    upsilon_omega_sigma.template head<2>() =
        mat_omega.col(2).template head<2>();
    upsilon_omega_sigma.template tail<2>() =
        RxSo2<Scalar>::vee(mat_omega.template topLeftCorner<2, 2>());
    return upsilon_omega_sigma;
  }

 protected:
  RxSo2Member rxso2_;              // NOLINT
  TranslationMember translation_;  // NOLINT
};

template <class TScalar, int kOptions>
SOPHUS_FUNC Sim2<TScalar, kOptions>::Sim2()
    : translation_(TranslationMember::Zero()) {
  static_assert(
      std::is_standard_layout<Sim2>::value,
      "Assume standard layout for the use of offsetof check below.");
  static_assert(
      offsetof(Sim2, rxso2_) + sizeof(Scalar) * RxSo2<Scalar>::kNumParameters ==
          offsetof(Sim2, translation_),
      "This class assumes packed storage and hence will only work "
      "correctly depending on the compiler (options) - in "
      "particular when using [this->data(), this-data() + "
      "kNumParameters] to access the raw data in a contiguous fashion.");
}

}  // namespace sophus

namespace Eigen {  // NOLINT

/// Specialization of Eigen::Map for ``Sim2``; derived from Sim2Base.
///
/// Allows us to wrap Sim2 objects around POD array.
template <class TScalar, int kOptions>
class Map<sophus::Sim2<TScalar>, kOptions>
    : public sophus::Sim2Base<Map<sophus::Sim2<TScalar>, kOptions>> {
 public:
  using Base = sophus::Sim2Base<Map<sophus::Sim2<TScalar>, kOptions>>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar* coeffs)
      : rxso2_(coeffs),
        translation_(coeffs + sophus::RxSo2<Scalar>::kNumParameters) {}

  /// Mutator of RxSo2
  ///
  SOPHUS_FUNC Map<sophus::RxSo2<Scalar>, kOptions>& rxso2() { return rxso2_; }

  /// Accessor of RxSo2
  ///
  SOPHUS_FUNC [[nodiscard]] Map<sophus::RxSo2<Scalar>, kOptions> const& rxso2()
      const {
    return rxso2_;
  }

  /// Mutator of translation vector
  ///
  SOPHUS_FUNC Map<Eigen::Vector2<Scalar>, kOptions>& translation() {
    return translation_;
  }

  /// Accessor of translation vector
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector2<Scalar>, kOptions> const&
  translation() const {
    return translation_;
  }

 protected:
  Map<sophus::RxSo2<Scalar>, kOptions> rxso2_;         // NOLINT
  Map<Eigen::Vector2<Scalar>, kOptions> translation_;  // NOLINT
};

/// Specialization of Eigen::Map for ``Sim2 const``; derived from Sim2Base.
///
/// Allows us to wrap RxSo2 objects around POD array.
template <class TScalar, int kOptions>
class Map<sophus::Sim2<TScalar> const, kOptions>
    : public sophus::Sim2Base<Map<sophus::Sim2<TScalar> const, kOptions>> {
 public:
  using Base = sophus::Sim2Base<Map<sophus::Sim2<TScalar> const, kOptions>>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar const* coeffs)
      : rxso2_(coeffs),
        translation_(coeffs + sophus::RxSo2<Scalar>::kNumParameters) {}

  /// Accessor of RxSo2
  ///
  SOPHUS_FUNC [[nodiscard]] Map<sophus::RxSo2<Scalar> const, kOptions> const&
  rxso2() const {
    return rxso2_;
  }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector2<Scalar> const, kOptions> const&
  translation() const {
    return translation_;
  }

 protected:
  Map<sophus::RxSo2<Scalar> const, kOptions> rxso2_;         // NOLINT
  Map<Eigen::Vector2<Scalar> const, kOptions> translation_;  // NOLINT
};
}  // namespace Eigen
