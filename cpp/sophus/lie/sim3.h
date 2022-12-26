// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Similarity group Sim(3) - scaling, rotation and translation in 3d.

#pragma once

#include "sophus/lie/details/sim_impl.h"
#include "sophus/lie/rxso3.h"

namespace sophus {
template <class TScalar, int kOptions = 0>
class Sim3;
using Sim3F64 = Sim3<double>;
using Sim3F32 = Sim3<float>;

/* [[deprecated]] */ using Sim3d = Sim3F64;
/* [[deprecated]] */ using Sim3f = Sim3F32;
}  // namespace sophus

namespace Eigen {  // NOLINT
namespace internal {

template <class TScalar, int kOptions>
struct traits<sophus::Sim3<TScalar, kOptions>> {
  using Scalar = TScalar;
  using TranslationType = Eigen::Matrix<Scalar, 3, 1, kOptions>;
  using RxSo3Type = sophus::RxSo3<Scalar, kOptions>;
};

template <class TScalar, int kOptions>
struct traits<Map<sophus::Sim3<TScalar>, kOptions>>
    : traits<sophus::Sim3<TScalar, kOptions>> {
  using Scalar = TScalar;
  using TranslationType = Map<Eigen::Vector3<Scalar>, kOptions>;
  using RxSo3Type = Map<sophus::RxSo3<Scalar>, kOptions>;
};

template <class TScalar, int kOptions>
struct traits<Map<sophus::Sim3<TScalar> const, kOptions>>
    : traits<sophus::Sim3<TScalar, kOptions> const> {
  using Scalar = TScalar;
  using TranslationType = Map<Eigen::Vector3<Scalar> const, kOptions>;
  using RxSo3Type = Map<sophus::RxSo3<Scalar> const, kOptions>;
};
}  // namespace internal
}  // namespace Eigen

namespace sophus {

/// Sim3 base type - implements Sim3 class but is storage agnostic.
///
/// Sim(3) is the group of rotations  and translation and scaling in 3d. It is
/// the semi-direct product of R+xSO(3) and the 3d Euclidean vector space.  The
/// class is represented using a composition of RxSo3  for scaling plus
/// rotation and a 3-vector for translation.
///
/// Sim(3) is neither compact, nor a commutative group.
///
///  - 4x4 Eigen::Matrix representation:
///
/// ```
///   | s*R t |
///   |  o  1 |
/// ```
///
/// where ``R`` is a 3x3 rotation matrix, ``s`` a positive scale factor,
/// ``t`` a translation 3-vector and ``o`` a 3-column vector of zeros.
///
///  - Tangent 7-vector: [upsilon, omega, sigma],
///
/// where ``upsilon`` is the translational velocity 3-vector and ``omega`` the
/// rotational velocity 3-vector, and ``sigma = log(s)``.
///
///  - Internal 7-representation: [t0, t1, t2, qi0, qi1, qi2, qr],
///
/// with ``t0, t1, t2`` are the translational components, and ``qi0, qi1, q2``
/// the imaginary vector part and ``qr1`` the real part of a non-zero
/// quaternion. Here the scale ``s`` is equal to the squared norm of the
/// quaternion ``s = |q|^2``.
///
/// See RxSo3 for more details of the scaling + rotation representation in 3d.
///
template <class TDerived>
class Sim3Base {
 public:
  using Scalar = typename Eigen::internal::traits<TDerived>::Scalar;
  using TranslationType =
      typename Eigen::internal::traits<TDerived>::TranslationType;
  using RxSo3Type = typename Eigen::internal::traits<TDerived>::RxSo3Type;
  using QuaternionType = typename RxSo3Type::QuaternionType;

  /// Degrees of freedom of manifold, number of dimensions in tangent space
  /// (three for translation, three for rotation and one for scaling).
  static int constexpr kDoF = 7;
  /// Number of internal parameters used (4-tuple for quaternion, three for
  /// translation).
  static int constexpr kNumParameters = 7;
  /// Group transformations are 4x4 matrices.
  static int constexpr kMatrixDim = 4;
  /// Points are 3-dimensional
  static int constexpr kPointDim = 3;
  using Transformation = Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim>;
  using Point = Eigen::Vector3<Scalar>;
  using HomogeneousPoint = Eigen::Vector4<Scalar>;
  using Line = Eigen::ParametrizedLine<Scalar, 3>;
  using Hyperplane = Eigen::Hyperplane<Scalar, 3>;
  using Tangent = Eigen::Vector<Scalar, kDoF>;
  using Adjoint = Eigen::Matrix<Scalar, kDoF, kDoF>;

  /// For binary operations the return type is determined with the
  /// ScalarBinaryOpTraits feature of Eigen. This allows mixing concrete and Map
  /// types, as well as other compatible scalar types such as Ceres::Jet and
  /// double scalars with Sim3 operations.
  template <class TOtherDerived>
  using ReturnScalar = typename Eigen::
      ScalarBinaryOpTraits<Scalar, typename TOtherDerived::Scalar>::ReturnType;

  template <class TOtherDerived>
  using Sim3Product = Sim3<ReturnScalar<TOtherDerived>>;

  template <class TPointDerived>
  using PointProduct = Eigen::Vector3<ReturnScalar<TPointDerived>>;

  template <class THPointDerived>
  using HomogeneousPointProduct = Eigen::Vector4<ReturnScalar<THPointDerived>>;

  /// Adjoint transformation
  ///
  /// This function return the adjoint transformation ``Ad`` of the group
  /// element ``A`` such that for all ``x`` it holds that
  /// ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  ///
  SOPHUS_FUNC [[nodiscard]] Adjoint adj() const {
    Eigen::Matrix3<Scalar> const r = rxso3().rotationMatrix();
    Adjoint res;
    res.setZero();
    res.template block<3, 3>(0, 0) = rxso3().matrix();
    res.template block<3, 3>(0, 3) = So3<Scalar>::hat(translation()) * r;
    res.template block<3, 1>(0, 6) = -translation();

    res.template block<3, 3>(3, 3) = r;

    res(6, 6) = Scalar(1);
    return res;
  }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class TNewScalarType>
  SOPHUS_FUNC [[nodiscard]] Sim3<TNewScalarType> cast() const {
    return Sim3<TNewScalarType>(
        rxso3().template cast<TNewScalarType>(),
        translation().template cast<TNewScalarType>());
  }

  /// Returns group inverse.
  ///
  SOPHUS_FUNC [[nodiscard]] Sim3<Scalar> inverse() const {
    RxSo3<Scalar> inv_r = rxso3().inverse();
    return Sim3<Scalar>(inv_r, inv_r * (translation() * Scalar(-1)));
  }

  /// Logarithmic map
  ///
  /// Computes the logarithm, the inverse of the group exponential which maps
  /// element of the group (rigid body transformations) to elements of the
  /// tangent space (twist).
  ///
  /// To be specific, this function computes ``vee(logmat(.))`` with
  /// ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  /// of Sim(3).
  ///
  SOPHUS_FUNC [[nodiscard]] Tangent log() const {
    // The derivation of the closed-form Sim(3) logarithm for is done
    // analogously to the closed-form solution of the SE(3) logarithm, see
    // J. Gallier, D. Xu, "Computing exponentials of skew symmetric matrices
    // and logarithms of orthogonal matrices", IJRA 2002.
    // https:///pdfs.semanticscholar.org/cfe3/e4b39de63c8cabd89bf3feff7f5449fc981d.pdf
    // (Sec. 6., pp. 8)
    Tangent res;
    auto omega_sigma_and_theta = rxso3().logAndTheta();
    Eigen::Vector3<Scalar> const omega =
        omega_sigma_and_theta.tangent.template head<3>();
    Scalar sigma = omega_sigma_and_theta.tangent[3];
    Eigen::Matrix3<Scalar> const mat_omega = So3<Scalar>::hat(omega);
    Eigen::Matrix3<Scalar> const w_inv = details::calcWInv<Scalar, 3>(
        mat_omega, omega_sigma_and_theta.theta, sigma, scale());

    res.segment(0, 3) = w_inv * translation();
    res.segment(3, 3) = omega;
    res[6] = sigma;
    return res;
  }

  /// Returns 4x4 matrix representation of the instance.
  ///
  /// It has the following form:
  ///
  ///     | s*R t |
  ///     |  o  1 |
  ///
  /// where ``R`` is a 3x3 rotation matrix, ``s`` a scale factor, ``t`` a
  /// translation 3-vector and ``o`` a 3-column vector of zeros.
  ///
  SOPHUS_FUNC [[nodiscard]] Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.template topLeftCorner<3, 4>() = matrix3x4();
    homogenious_matrix.row(3) =
        Eigen::Matrix<Scalar, 4, 1>(Scalar(0), Scalar(0), Scalar(0), Scalar(1));
    return homogenious_matrix;
  }

  /// Returns the significant first three rows of the matrix above.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, 3, 4> matrix3x4() const {
    Eigen::Matrix<Scalar, 3, 4> matrix;
    matrix.template topLeftCorner<3, 3>() = rxso3().matrix();
    matrix.col(3) = translation();
    return matrix;
  }

  /// Assignment-like operator from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Sim3Base<TDerived>& operator=(
      Sim3Base<TOtherDerived> const& other) {
    rxso3() = other.rxso3();
    translation() = other.translation();
    return *this;
  }

  /// Group multiplication, which is rotation plus scaling concatenation.
  ///
  /// Note: That scaling is calculated with saturation. See RxSo3 for
  /// details.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Sim3Product<TOtherDerived> operator*(
      Sim3Base<TOtherDerived> const& other) const {
    return Sim3Product<TOtherDerived>(
        rxso3() * other.rxso3(), translation() + rxso3() * other.translation());
  }

  /// Group action on 3-points.
  ///
  /// This function rotates, scales and translates a three dimensional point
  /// ``p`` by the Sim(3) element ``(bar_sR_foo, t_bar)`` (= similarity
  /// transformation):
  ///
  ///   ``p_bar = bar_sR_foo * p_foo + t_bar``.
  ///
  template <
      typename TPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<TPointDerived, 3>::value>::type>
  SOPHUS_FUNC PointProduct<TPointDerived> operator*(
      Eigen::MatrixBase<TPointDerived> const& p) const {
    return rxso3() * p + translation();
  }

  /// Group action on homogeneous 3-points. See above for more details.
  ///
  template <
      typename THPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<THPointDerived, 4>::value>::type>
  SOPHUS_FUNC HomogeneousPointProduct<THPointDerived> operator*(
      Eigen::MatrixBase<THPointDerived> const& p) const {
    PointProduct<THPointDerived> const tp =
        rxso3() * p.template head<3>() + p(3) * translation();
    return HomogeneousPointProduct<THPointDerived>(tp(0), tp(1), tp(2), p(3));
  }

  /// Group action on lines.
  ///
  /// This function rotates, scales and translates a parametrized line
  /// ``l(t) = o + t * d`` by the Sim(3) element:
  ///
  /// Origin ``o`` is rotated, scaled and translated
  /// Direction ``d`` is rotated
  ///
  SOPHUS_FUNC Line operator*(Line const& l) const {
    Line rotated_line = rxso3() * l;
    return Line(
        rotated_line.origin() + translation(), rotated_line.direction());
  }

  /// Group action on planes.
  ///
  /// This function rotates and translates a plane
  /// ``n.x + d = 0`` by the Sim(3) element:
  ///
  /// Normal vector ``n`` is rotated
  /// Offset ``d`` is adjusted for scale and translation
  ///
  SOPHUS_FUNC Hyperplane operator*(Hyperplane const& p) const {
    Hyperplane const rotated = rxso3() * p;
    return Hyperplane(
        rotated.normal(),
        rotated.offset() - translation().dot(rotated.normal()));
  }

  /// In-place group multiplication. This method is only valid if the return
  /// type of the multiplication is compatible with this So3's Scalar type.
  ///
  template <
      typename TOtherDerived,
      typename = typename std::enable_if<
          std::is_same<Scalar, ReturnScalar<TOtherDerived>>::value>::type>
  SOPHUS_FUNC Sim3Base<TDerived>& operator*=(
      Sim3Base<TOtherDerived> const& other) {
    *static_cast<TDerived*>(this) = *this * other;
    return *this;
  }

  /// Returns derivative of  this * Sim3::exp(x) w.r.t. x at x = 0
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kNumParameters, kDoF>
  dxThisMulExpXAt0() const {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> j;
    j.template block<4, 3>(0, 0).setZero();
    j.template block<4, 4>(0, 3) = rxso3().dxThisMulExpXAt0();
    j.template block<3, 3>(4, 0) = rxso3().matrix();
    j.template block<3, 4>(4, 3).setZero();

    return j;
  }

  /// Returns derivative of log(this^{-1} * x) by x at x=this.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kDoF, kNumParameters>
  dxLogThisInvTimesXAtThis() const {
    Eigen::Matrix<Scalar, kDoF, kNumParameters> j;
    j.template block<3, 4>(0, 0).setZero();
    j.template block<3, 3>(0, 4) = rxso3().inverse().matrix();
    j.template block<4, 4>(3, 0) = rxso3().dxLogThisInvTimesXAtThis();
    j.template block<4, 3>(3, 4).setZero();
    return j;
  }

  /// Returns internal parameters of Sim(3).
  ///
  /// It returns (q.imag[0], q.imag[1], q.imag[2], q.real, t[0], t[1], t[2]),
  /// with q being the quaternion, t the translation 3-vector.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Vector<Scalar, kNumParameters> params()
      const {
    Eigen::Vector<Scalar, kNumParameters> p;
    p << rxso3().params(), translation();
    return p;
  }

  /// Setter of non-zero quaternion.
  ///
  /// Precondition: ``quat`` must not be close to zero.
  ///
  SOPHUS_FUNC void setQuaternion(Eigen::Quaternion<Scalar> const& quat) {
    rxso3().setQuaternion(quat);
  }

  /// Accessor of quaternion.
  ///
  SOPHUS_FUNC [[nodiscard]] QuaternionType const& quaternion() const {
    return rxso3().quaternion();
  }

  /// Returns Rotation matrix
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix3<Scalar> rotationMatrix() const {
    return rxso3().rotationMatrix();
  }

  /// Mutator of So3 group.
  ///
  SOPHUS_FUNC RxSo3Type& rxso3() {
    return static_cast<TDerived*>(this)->rxso3();
  }

  /// Accessor of So3 group.
  ///
  SOPHUS_FUNC [[nodiscard]] RxSo3Type const& rxso3() const {
    return static_cast<TDerived const*>(this)->rxso3();
  }

  /// Returns scale.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar scale() const { return rxso3().scale(); }

  /// Setter of quaternion using rotation matrix ``R``, leaves scale as is.
  ///
  SOPHUS_FUNC void setRotationMatrix(Eigen::Matrix3<Scalar>& r) {
    rxso3().setRotationMatrix(r);
  }

  /// Sets scale and leaves rotation as is.
  ///
  /// Note: This function as a significant computational cost, since it has to
  /// call the square root twice.
  ///
  SOPHUS_FUNC void setScale(Scalar const& scale) { rxso3().setScale(scale); }

  /// Setter of quaternion using scaled rotation matrix ``sR``.
  ///
  /// Precondition: The 3x3 matrix must be "scaled orthogonal"
  ///               and have a positive determinant.
  ///
  SOPHUS_FUNC void setScaledRotationMatrix(Eigen::Matrix3<Scalar> const& s_r) {
    rxso3().setScaledRotationMatrix(s_r);
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

/// Sim3 using default storage; derived from Sim3Base.
template <class TScalar, int kOptions>
class Sim3 : public Sim3Base<Sim3<TScalar, kOptions>> {
 public:
  using Base = Sim3Base<Sim3<TScalar, kOptions>>;
  static int constexpr kDoF = Base::kDoF;
  static int constexpr kNumParameters = Base::kNumParameters;

  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using RxSo3Member = RxSo3<Scalar, kOptions>;
  using TranslationMember = Eigen::Matrix<Scalar, 3, 1, kOptions>;

  using Base::operator=;

  /// Define copy-assignment operator explicitly. The definition of
  /// implicit copy assignment operator is deprecated in presence of a
  /// user-declared copy constructor (-Wdeprecated-copy in clang >= 13).
  SOPHUS_FUNC Sim3& operator=(Sim3 const& other) = default;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Default constructor initializes similarity transform to the identity.
  ///
  SOPHUS_FUNC Sim3();

  /// Copy constructor
  ///
  SOPHUS_FUNC Sim3(Sim3 const& other) = default;

  /// Copy-like constructor from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Sim3(Sim3Base<TOtherDerived> const& other)
      : rxso3_(other.rxso3()), translation_(other.translation()) {
    static_assert(
        std::is_same<typename TOtherDerived::Scalar, Scalar>::value,
        "must be same Scalar type");
  }

  /// Constructor from RxSo3 and translation vector
  ///
  template <class TOtherDerived, class TD>
  SOPHUS_FUNC explicit Sim3(
      RxSo3Base<TOtherDerived> const& rxso3,
      Eigen::MatrixBase<TD> const& translation)
      : rxso3_(rxso3), translation_(translation) {
    static_assert(
        std::is_same<typename TOtherDerived::Scalar, Scalar>::value,
        "must be same Scalar type");
    static_assert(
        std::is_same<typename TD::Scalar, Scalar>::value,
        "must be same Scalar type");
  }

  /// Constructor from quaternion and translation vector.
  ///
  /// Precondition: quaternion must not be close to zero.
  ///
  template <class TD1T, class TD2T>
  SOPHUS_FUNC explicit Sim3(
      Eigen::QuaternionBase<TD1T> const& quaternion,
      Eigen::MatrixBase<TD2T> const& translation)
      : rxso3_(quaternion), translation_(translation) {
    static_assert(
        std::is_same<typename TD1T::Scalar, Scalar>::value,
        "must be same Scalar type");
    static_assert(
        std::is_same<typename TD2T::Scalar, Scalar>::value,
        "must be same Scalar type");
  }

  /// Constructor from scale factor, unit quaternion, and translation vector.
  ///
  /// Precondition: quaternion must not be close to zero.
  ///
  template <class TD1T, class TD2T>
  SOPHUS_FUNC explicit Sim3(
      Scalar const& scale,
      Eigen::QuaternionBase<TD1T> const& unit_quaternion,
      Eigen::MatrixBase<TD2T> const& translation)
      : Sim3(RxSo3<Scalar>(scale, unit_quaternion), translation) {}

  /// Constructor from 4x4 matrix
  ///
  /// Precondition: Top-left 3x3 matrix needs to be "scaled-orthogonal" with
  ///               positive determinant. The last row must be ``(0, 0, 0, 1)``.
  ///
  SOPHUS_FUNC explicit Sim3(Eigen::Matrix<Scalar, 4, 4> const& t)
      : rxso3_(t.template topLeftCorner<3, 3>()),
        translation_(t.template block<3, 1>(0, 3)) {}

  /// This provides unsafe read/write access to internal data. Sim(3) is
  /// represented by an Eigen::Quaternion (four parameters) and a 3-vector. When
  /// using direct write access, the user needs to take care of that the
  /// quaternion is not set close to zero.
  ///
  SOPHUS_FUNC Scalar* data() {
    // rxso3_ and translation_ are laid out sequentially with no padding
    return rxso3_.data();
  }

  /// Const version of data() above.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar const* data() const {
    // rxso3_ and translation_ are laid out sequentially with no padding
    return rxso3_.data();
  }

  /// Accessor of RxSo3
  ///
  SOPHUS_FUNC RxSo3Member& rxso3() { return rxso3_; }

  /// Mutator of RxSo3
  ///
  SOPHUS_FUNC [[nodiscard]] RxSo3Member const& rxso3() const { return rxso3_; }

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
    Eigen::Matrix<Scalar, kNumParameters, kDoF> dx;
    dx.template block<4, 3>(0, 0).setZero();
    dx.template block<4, 4>(0, 3) = RxSo3<Scalar>::dxExpXAt0();
    dx.template block<3, 3>(4, 0).setIdentity();
    dx.template block<3, 4>(4, 3).setZero();
    return dx;
  }

  /// Returns derivative of exp(x) wrt. x.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpX(
      Tangent const& vec_a) {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> dx;

    static Eigen::Matrix3<Scalar> const kI = Eigen::Matrix3<Scalar>::Identity();
    Eigen::Vector3<Scalar> const vec_omega = vec_a.template segment<3>(3);
    Eigen::Vector3<Scalar> const vec_upsilon = vec_a.template head<3>();
    Scalar const sigma = vec_a[6];
    Scalar const theta = vec_omega.norm();

    Eigen::Matrix3<Scalar> const mat_omega = So3<Scalar>::hat(vec_omega);
    Eigen::Matrix3<Scalar> const mat_omega2 = mat_omega * mat_omega;
    Eigen::Vector3<Scalar> theta_domega;
    if (theta < kEpsilon<Scalar>) {
      theta_domega = Eigen::Vector3<Scalar>::Zero();
    } else {
      theta_domega = vec_omega / theta;
    }
    static Eigen::Matrix3<Scalar> const kOmegaDomega[3] = {
        So3<Scalar>::hat(Eigen::Vector3<Scalar>::Unit(0)),
        So3<Scalar>::hat(Eigen::Vector3<Scalar>::Unit(1)),
        So3<Scalar>::hat(Eigen::Vector3<Scalar>::Unit(2))};

    Eigen::Matrix3<Scalar> const omega2_domega[3] = {
        kOmegaDomega[0] * mat_omega + mat_omega * kOmegaDomega[0],
        kOmegaDomega[1] * mat_omega + mat_omega * kOmegaDomega[1],
        kOmegaDomega[2] * mat_omega + mat_omega * kOmegaDomega[2]};

    Eigen::Matrix3<Scalar> const w =
        details::calcW<Scalar, 3>(mat_omega, theta, sigma);

    dx.template block<4, 3>(0, 0).setZero();
    dx.template block<4, 4>(0, 3) =
        RxSo3<Scalar>::dxExpX(vec_a.template tail<4>());
    dx.template block<3, 4>(4, 3).setZero();
    dx.template block<3, 3>(4, 0) = w;

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

    for (int i = 0; i < 3; ++i) {
      dx.template block<3, 1>(4, 3 + i) =
          (a_dtheta * theta_domega[i] * mat_omega + a * kOmegaDomega[i] +
           b_dtheta * theta_domega[i] * mat_omega2 + b * omega2_domega[i]) *
          vec_upsilon;
    }

    dx.template block<3, 1>(4, 6) =
        (a_dsigma * mat_omega + b_dsigma * mat_omega2 + c_dsigma * kI) *
        vec_upsilon;

    return dx;
  }

  /// Returns derivative of exp(x) * p wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, 3, kDoF> dxExpXTimesPointAt0(
      Point const& point) {
    Eigen::Matrix<Scalar, 3, kDoF> dx;
    dx << Eigen::Matrix3<Scalar>::Identity(),
        sophus::RxSo3<Scalar>::dxExpXTimesPointAt0(point);
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
  /// This functions takes in an element of tangent space and returns the
  /// corresponding element of the group Sim(3).
  ///
  /// The first three components of ``a`` represent the translational part
  /// ``upsilon`` in the tangent space of Sim(3), the following three components
  /// of ``a`` represents the rotation vector ``omega`` and the final component
  /// represents the logarithm of the scaling factor ``sigma``.
  /// To be more specific, this function computes ``expmat(hat(a))`` with
  /// ``expmat(.)`` being the matrix exponential and ``hat(.)`` the hat-operator
  /// of Sim(3), see below.
  ///
  SOPHUS_FUNC static Sim3<Scalar> exp(Tangent const& vec_a) {
    // For the derivation of the exponential map of Sim(3) see
    // H. Strasdat, "Local Accuracy and Global Consistency for Efficient Visual
    // SLAM", PhD thesis, 2012.
    // http:///hauke.strasdat.net/files/strasdat_thesis_2012.pdf (A.5, pp. 186)
    Eigen::Vector3<Scalar> const vec_upsilon = vec_a.segment(0, 3);
    Eigen::Vector3<Scalar> const vec_omega = vec_a.segment(3, 3);
    Scalar const sigma = vec_a[6];
    auto rxso3_and_theta = RxSo3<Scalar>::expAndTheta(vec_a.template tail<4>());
    Eigen::Matrix3<Scalar> const mat_omega = So3<Scalar>::hat(vec_omega);

    Eigen::Matrix3<Scalar> const w =
        details::calcW<Scalar, 3>(mat_omega, rxso3_and_theta.theta, sigma);
    return Sim3<Scalar>(rxso3_and_theta.rxso3, w * vec_upsilon);
  }

  /// Returns the ith infinitesimal generators of Sim(3).
  ///
  /// The infinitesimal generators of Sim(3) are:
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
  ///
  ///         |  0  0  0  0 |
  ///   G_3 = |  0  0 -1  0 |
  ///         |  0  1  0  0 |
  ///         |  0  0  0  0 |
  ///
  ///         |  0  0  1  0 |
  ///   G_4 = |  0  0  0  0 |
  ///         | -1  0  0  0 |
  ///         |  0  0  0  0 |
  ///
  ///         |  0 -1  0  0 |
  ///   G_5 = |  1  0  0  0 |
  ///         |  0  0  0  0 |
  ///         |  0  0  0  0 |
  ///
  ///         |  1  0  0  0 |
  ///   G_6 = |  0  1  0  0 |
  ///         |  0  0  1  0 |
  ///         |  0  0  0  0 |
  /// ```
  ///
  /// Precondition: ``i`` must be in [0, 6].
  ///
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ASSERT(i >= 0 || i <= 6, "i should be in range [0,6].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  /// hat-operator
  ///
  /// It takes in the 7-vector representation and returns the corresponding
  /// matrix representation of Lie algebra element.
  ///
  /// Formally, the hat()-operator of Sim(3) is defined as
  ///
  ///   ``hat(.): R^7 -> R^{4x4},  hat(a) = sum_i a_i * G_i``  (for i=0,...,6)
  ///
  /// with ``G_i`` being the ith infinitesimal generator of Sim(3).
  ///
  /// The corresponding inverse is the vee()-operator, see below.
  ///
  SOPHUS_FUNC static Transformation hat(Tangent const& vec_a) {
    Transformation vec_omega;
    vec_omega.template topLeftCorner<3, 3>() =
        RxSo3<Scalar>::hat(vec_a.template tail<4>());
    vec_omega.col(3).template head<3>() = vec_a.template head<3>();
    vec_omega.row(3).setZero();
    return vec_omega;
  }

  /// Lie bracket
  ///
  /// It computes the Lie bracket of Sim(3). To be more specific, it computes
  ///
  ///   ``[omega_1, omega_2]_sim3 := vee([hat(omega_1), hat(omega_2)])``
  ///
  /// with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.)`` the
  /// hat()-operator and ``vee(.)`` the vee()-operator of Sim(3).
  ///
  SOPHUS_FUNC static Tangent lieBracket(
      Tangent const& vec_a, Tangent const& vec_b) {
    Eigen::Vector3<Scalar> const vec_upsilon1 = vec_a.template head<3>();
    Eigen::Vector3<Scalar> const vec_upsilon2 = vec_b.template head<3>();
    Eigen::Vector3<Scalar> const vec_omega1 = vec_a.template segment<3>(3);
    Eigen::Vector3<Scalar> const vec_omega2 = vec_b.template segment<3>(3);
    Scalar sigma1 = vec_a[6];
    Scalar sigma2 = vec_b[6];

    Tangent res;
    res.template head<3>() = So3<Scalar>::hat(vec_omega1) * vec_upsilon2 +
                             So3<Scalar>::hat(vec_upsilon1) * vec_omega2 +
                             sigma1 * vec_upsilon2 - sigma2 * vec_upsilon1;
    res.template segment<3>(3) = vec_omega1.cross(vec_omega2);
    res[6] = Scalar(0);

    return res;
  }

  /// Draw uniform sample from Sim(3) manifold.
  ///
  /// Translations are drawn component-wise from the range [-1, 1].
  /// The scale factor is drawn uniformly in log2-space from [-1, 1],
  /// hence the scale is in [0.5, 2].
  ///
  template <class TUniformRandomBitGenerator>
  static Sim3 sampleUniform(TUniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    return Sim3(
        RxSo3<Scalar>::sampleUniform(generator),
        Eigen::Vector3<Scalar>(
            uniform(generator), uniform(generator), uniform(generator)));
  }

  /// vee-operator
  ///
  /// It takes the 4x4-matrix representation ``Omega`` and maps it to the
  /// corresponding 7-vector representation of Lie algebra.
  ///
  /// This is the inverse of the hat()-operator, see above.
  ///
  /// Precondition: ``Omega`` must have the following structure:
  ///
  ///                |  g -f  e  a |
  ///                |  f  g -d  b |
  ///                | -e  d  g  c |
  ///                |  0  0  0  0 |
  ///
  SOPHUS_FUNC static Tangent vee(Transformation const& mat_omega) {
    Tangent vec_upsilon_omega_sigma;
    vec_upsilon_omega_sigma.template head<3>() =
        mat_omega.col(3).template head<3>();
    vec_upsilon_omega_sigma.template tail<4>() =
        RxSo3<Scalar>::vee(mat_omega.template topLeftCorner<3, 3>());
    return vec_upsilon_omega_sigma;
  }

 protected:
  RxSo3Member rxso3_;              // NOLINT
  TranslationMember translation_;  // NOLINT
};

template <class TScalar, int kOptions>
SOPHUS_FUNC Sim3<TScalar, kOptions>::Sim3()
    : translation_(TranslationMember::Zero()) {
  static_assert(
      std::is_standard_layout<Sim3>::value,
      "Assume standard layout for the use of offsetof check below.");
  static_assert(
      offsetof(Sim3, rxso3_) + sizeof(Scalar) * RxSo3<Scalar>::kNumParameters ==
          offsetof(Sim3, translation_),
      "This class assumes packed storage and hence will only work "
      "correctly depending on the compiler (options) - in "
      "particular when using [this->data(), this-data() + "
      "kNumParameters] to access the raw data in a contiguous fashion.");
}

}  // namespace sophus

namespace Eigen {  // NOLINT

/// Specialization of Eigen::Map for ``Sim3``; derived from Sim3Base.
///
/// Allows us to wrap Sim3 objects around POD array.
template <class TScalar, int kOptions>
class Map<sophus::Sim3<TScalar>, kOptions>
    : public sophus::Sim3Base<Map<sophus::Sim3<TScalar>, kOptions>> {
 public:
  using Base = sophus::Sim3Base<Map<sophus::Sim3<TScalar>, kOptions>>;
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
      : rxso3_(coeffs),
        translation_(coeffs + sophus::RxSo3<Scalar>::kNumParameters) {}

  /// Mutator of RxSo3
  ///
  SOPHUS_FUNC Map<sophus::RxSo3<Scalar>, kOptions>& rxso3() { return rxso3_; }

  /// Accessor of RxSo3
  ///
  SOPHUS_FUNC [[nodiscard]] Map<sophus::RxSo3<Scalar>, kOptions> const& rxso3()
      const {
    return rxso3_;
  }

  /// Mutator of translation vector
  ///
  SOPHUS_FUNC Map<Eigen::Vector3<Scalar>, kOptions>& translation() {
    return translation_;
  }

  /// Accessor of translation vector
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector3<Scalar>, kOptions> const&
  translation() const {
    return translation_;
  }

 protected:
  Map<sophus::RxSo3<Scalar>, kOptions> rxso3_;         // NOLINT
  Map<Eigen::Vector3<Scalar>, kOptions> translation_;  // NOLINT
};

/// Specialization of Eigen::Map for ``Sim3 const``; derived from Sim3Base.
///
/// Allows us to wrap RxSo3 objects around POD array.
template <class TScalar, int kOptions>
class Map<sophus::Sim3<TScalar> const, kOptions>
    : public sophus::Sim3Base<Map<sophus::Sim3<TScalar> const, kOptions>> {
  using Base = sophus::Sim3Base<Map<sophus::Sim3<TScalar> const, kOptions>>;

 public:
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar const* coeffs)
      : rxso3_(coeffs),
        translation_(coeffs + sophus::RxSo3<Scalar>::kNumParameters) {}

  /// Accessor of RxSo3
  ///
  SOPHUS_FUNC [[nodiscard]] Map<sophus::RxSo3<Scalar> const, kOptions> const&
  rxso3() const {
    return rxso3_;
  }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector3<Scalar> const, kOptions> const&
  translation() const {
    return translation_;
  }

 protected:
  Map<sophus::RxSo3<Scalar> const, kOptions> rxso3_;         // NOLINT
  Map<Eigen::Vector3<Scalar> const, kOptions> translation_;  // NOLINT
};
}  // namespace Eigen
