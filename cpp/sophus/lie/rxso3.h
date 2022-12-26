// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Direct product R X SO(3) - rotation and scaling in 3d.

#pragma once

#include "so3.h"

namespace sophus {
template <class TScalar, int kOptions = 0>
class RxSo3;
using RxSo3F64 = RxSo3<double>;
using RxSo3F32 = RxSo3<float>;

template <class TScalar, int kOptions = 0>
/* [[deprecated]] */ using RxSO3 = RxSo3<TScalar, kOptions>;
/* [[deprecated]] */ using RxSO3d = RxSo3F64;
/* [[deprecated]] */ using RxSO3f = RxSo3F32;
}  // namespace sophus

namespace Eigen {  // NOLINT
namespace internal {

template <class TScalar, int kOptionsT>
struct traits<sophus::RxSo3<TScalar, kOptionsT>> {
  static int constexpr kOptions = kOptionsT;
  using Scalar = TScalar;
  using QuaternionType = Eigen::Quaternion<Scalar, kOptions>;
};

template <class TScalar, int kOptionsT>
struct traits<Map<sophus::RxSo3<TScalar>, kOptionsT>>
    : traits<sophus::RxSo3<TScalar, kOptionsT>> {
  static int constexpr kOptions = kOptionsT;
  using Scalar = TScalar;
  using QuaternionType = Map<Eigen::Quaternion<Scalar>, kOptions>;
};

template <class TScalar, int kOptionsT>
struct traits<Map<sophus::RxSo3<TScalar> const, kOptionsT>>
    : traits<sophus::RxSo3<TScalar, kOptionsT> const> {
  static int constexpr kOptions = kOptionsT;
  using Scalar = TScalar;
  using QuaternionType = Map<Eigen::Quaternion<Scalar> const, kOptions>;
};
}  // namespace internal
}  // namespace Eigen

namespace sophus {

/// RxSo3 base type - implements RxSo3 class but is storage agnostic
///
/// This class implements the group ``R+ x SO(3)``, the direct product of the
/// group of positive scalar 3x3 matrices (= isomorph to the positive
/// real numbers) and the three-dimensional special orthogonal group SO(3).
/// Geometrically, it is the group of rotation and scaling in three dimensions.
/// As a matrix groups, RxSo3 consists of matrices of the form ``s * R``
/// where ``R`` is an orthogonal matrix with ``det(R)=1`` and ``s > 0``
/// being a positive real number.
///
/// Internally, RxSo3 is represented by the group of non-zero quaternions.
/// In particular, the scale equals the squared(!) norm of the quaternion ``q``,
/// ``s = |q|^2``. This is a most compact representation since the degrees of
/// freedom (kDoF) of RxSo3 (=4) equals the number of internal parameters (=4).
///
/// This class has the explicit class invariant that the scale ``s`` is not
/// too close to either zero or infinity. Strictly speaking, it must hold that:
///
///   ``quaternion().squaredNorm() >= Constants::epsilon()`` and
///   ``1. / quaternion().squaredNorm() >= Constants::epsilon()``.
///
/// In order to obey this condition, group multiplication is implemented with
/// saturation such that a product always has a scale which is equal or greater
/// this threshold.
template <class TDerived>
class RxSo3Base {
 public:
  static int constexpr kOptions = Eigen::internal::traits<TDerived>::kOptions;
  using Scalar = typename Eigen::internal::traits<TDerived>::Scalar;
  using QuaternionType =
      typename Eigen::internal::traits<TDerived>::QuaternionType;
  using QuaternionTemporaryType = Eigen::Quaternion<Scalar, kOptions>;

  /// Degrees of freedom of manifold, number of dimensions in tangent space
  /// (three for rotation and one for scaling).
  static int constexpr kDoF = 4;
  /// Number of internal parameters used (quaternion is a 4-tuple).
  static int constexpr kNumParameters = 4;
  /// Group transformations are 3x3 matrices.
  static int constexpr kMatrixDim = 3;
  /// Points are 3-dimensional
  static int constexpr kPointDim = 3;
  using Transformation = Eigen::Matrix<Scalar, kMatrixDim, kMatrixDim>;
  using Point = Eigen::Vector3<Scalar>;
  using HomogeneousPoint = Eigen::Vector4<Scalar>;
  using Line = Eigen::ParametrizedLine<Scalar, 3>;
  using Hyperplane = Eigen::Hyperplane<Scalar, 3>;
  using Tangent = Eigen::Vector<Scalar, kDoF>;
  using Adjoint = Eigen::Matrix<Scalar, kDoF, kDoF>;

  struct TangentAndTheta {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Tangent tangent;
    Scalar theta;
  };

  /// For binary operations the return type is determined with the
  /// ScalarBinaryOpTraits feature of Eigen. This allows mixing concrete and Map
  /// types, as well as other compatible scalar types such as Ceres::Jet and
  /// double scalars with RxSo3 operations.
  template <class TOtherDerived>
  using ReturnScalar = typename Eigen::
      ScalarBinaryOpTraits<Scalar, typename TOtherDerived::Scalar>::ReturnType;

  template <class TOtherDerived>
  using RxSo3Product = RxSo3<ReturnScalar<TOtherDerived>>;

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
  /// For RxSO(3), it simply returns the rotation matrix corresponding to ``A``.
  ///
  SOPHUS_FUNC [[nodiscard]] Adjoint adj() const {
    Adjoint res;
    res.setIdentity();
    res.template topLeftCorner<3, 3>() = rotationMatrix();
    return res;
  }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class TNewScalarType>
  SOPHUS_FUNC [[nodiscard]] RxSo3<TNewScalarType> cast() const {
    return RxSo3<TNewScalarType>(quaternion().template cast<TNewScalarType>());
  }

  /// This provides unsafe read/write access to internal data. RxSO(3) is
  /// represented by an Eigen::Quaternion (four parameters). When using direct
  /// write access, the user needs to take care of that the quaternion is not
  /// set close to zero.
  ///
  /// Note: The first three Scalars represent the imaginary parts, while the
  /// forth Scalar represent the real part.
  ///
  SOPHUS_FUNC Scalar* data() { return quaternionNonconst().coeffs().data(); }

  /// Const version of data() above.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar const* data() const {
    return quaternion().coeffs().data();
  }

  /// Returns group inverse.
  ///
  SOPHUS_FUNC [[nodiscard]] RxSo3<Scalar> inverse() const {
    return RxSo3<Scalar>(quaternion().inverse());
  }

  /// Logarithmic map
  ///
  /// Computes the logarithm, the inverse of the group exponential which maps
  /// element of the group (scaled rotation matrices) to elements of the tangent
  /// space (rotation-vector plus logarithm of scale factor).
  ///
  /// To be specific, this function computes ``vee(logmat(.))`` with
  /// ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  /// of RxSo3.
  ///
  SOPHUS_FUNC [[nodiscard]] Tangent log() const {
    return logAndTheta().tangent;
  }

  /// As above, but also returns ``theta = |omega|``.
  ///
  SOPHUS_FUNC [[nodiscard]] TangentAndTheta logAndTheta() const {
    using std::log;

    Scalar scale = quaternion().squaredNorm();
    TangentAndTheta result;
    result.tangent[3] = log(scale);
    auto omega_and_theta = So3<Scalar>(quaternion()).logAndTheta();
    result.tangent.template head<3>() = omega_and_theta.tangent;
    result.theta = omega_and_theta.theta;
    return result;
  }
  /// Returns 3x3 matrix representation of the instance.
  ///
  /// For RxSo3, the matrix representation is an scaled orthogonal matrix ``sR``
  /// with ``det(R)=s^3``, thus a scaled rotation matrix ``R``  with scale
  /// ``s``.
  ///
  SOPHUS_FUNC [[nodiscard]] Transformation matrix() const {
    Transformation s_r;

    Scalar const vx_sq = quaternion().vec().x() * quaternion().vec().x();
    Scalar const vy_sq = quaternion().vec().y() * quaternion().vec().y();
    Scalar const vz_sq = quaternion().vec().z() * quaternion().vec().z();
    Scalar const w_sq = quaternion().w() * quaternion().w();
    Scalar const two_vx = Scalar(2) * quaternion().vec().x();
    Scalar const two_vy = Scalar(2) * quaternion().vec().y();
    Scalar const two_vz = Scalar(2) * quaternion().vec().z();
    Scalar const two_vx_vy = two_vx * quaternion().vec().y();
    Scalar const two_vx_vz = two_vx * quaternion().vec().z();
    Scalar const two_vx_w = two_vx * quaternion().w();
    Scalar const two_vy_vz = two_vy * quaternion().vec().z();
    Scalar const two_vy_w = two_vy * quaternion().w();
    Scalar const two_vz_w = two_vz * quaternion().w();

    s_r(0, 0) = vx_sq - vy_sq - vz_sq + w_sq;
    s_r(1, 0) = two_vx_vy + two_vz_w;
    s_r(2, 0) = two_vx_vz - two_vy_w;

    s_r(0, 1) = two_vx_vy - two_vz_w;
    s_r(1, 1) = -vx_sq + vy_sq - vz_sq + w_sq;
    s_r(2, 1) = two_vx_w + two_vy_vz;

    s_r(0, 2) = two_vx_vz + two_vy_w;
    s_r(1, 2) = -two_vx_w + two_vy_vz;
    s_r(2, 2) = -vx_sq - vy_sq + vz_sq + w_sq;
    return s_r;
  }

  /// Assignment-like operator from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC RxSo3Base<TDerived>& operator=(
      RxSo3Base<TOtherDerived> const& other) {
    quaternionNonconst() = other.quaternion();
    return *this;
  }

  /// Group multiplication, which is rotation concatenation and scale
  /// multiplication.
  ///
  /// Note: This function performs saturation for products close to zero in
  /// order to ensure the class invariant.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC RxSo3Product<TOtherDerived> operator*(
      RxSo3Base<TOtherDerived> const& other) const {
    using std::sqrt;
    using ResultT = ReturnScalar<TOtherDerived>;
    using QuaternionProductType =
        typename RxSo3Product<TOtherDerived>::QuaternionType;

    QuaternionProductType result_quaternion(
        sophus::So3<double>::quaternionProduct<QuaternionProductType>(
            quaternion(), other.quaternion()));

    ResultT scale = result_quaternion.squaredNorm();
    if (scale < kEpsilon<ResultT>) {
      FARM_ASSERT(scale > ResultT(0), "Scale must be greater zero.");
      /// Saturation to ensure class invariant.
      result_quaternion.normalize();
      result_quaternion.coeffs() *= sqrt(kEpsilonPlus<ResultT>);
    }
    if (scale > ResultT(1.) / kEpsilon<ResultT>) {
      result_quaternion.normalize();
      result_quaternion.coeffs() /= sqrt(kEpsilonPlus<ResultT>);
    }
    return RxSo3Product<TOtherDerived>(result_quaternion);
  }

  /// Group action on 3-points.
  ///
  /// This function rotates a 3 dimensional point ``p`` by the So3 element
  ///  ``bar_R_foo`` (= rotation matrix) and scales it by the scale factor
  ///  ``s``:
  ///
  ///   ``p_bar = s * (bar_R_foo * p_foo)``.
  ///
  template <
      typename TPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<TPointDerived, 3>::value>::type>
  SOPHUS_FUNC PointProduct<TPointDerived> operator*(
      Eigen::MatrixBase<TPointDerived> const& p) const {
    // Follows http:///eigen.tuxfamily.org/bz/show_bug.cgi?id=459
    Scalar scale = quaternion().squaredNorm();
    PointProduct<TPointDerived> two_vec_cross_p = quaternion().vec().cross(p);
    two_vec_cross_p += two_vec_cross_p;
    return scale * p + (quaternion().w() * two_vec_cross_p +
                        quaternion().vec().cross(two_vec_cross_p));
  }

  /// Group action on homogeneous 3-points. See above for more details.
  ///
  template <
      typename THPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<THPointDerived, 4>::value>::type>
  SOPHUS_FUNC HomogeneousPointProduct<THPointDerived> operator*(
      Eigen::MatrixBase<THPointDerived> const& p) const {
    auto const rsp = *this * p.template head<3>();
    return HomogeneousPointProduct<THPointDerived>(
        rsp(0), rsp(1), rsp(2), p(3));
  }

  /// Group action on lines.
  ///
  /// This function rotates a parametrized line ``l(t) = o + t * d`` by the So3
  /// element ans scales it by the scale factor:
  ///
  /// Origin ``o`` is rotated and scaled
  /// Direction ``d`` is rotated (preserving it's norm)
  ///
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line(
        (*this) * l.origin(),
        (*this) * l.direction() / quaternion().squaredNorm());
  }

  /// Group action on planes.
  ///
  /// This function rotates parametrized plane
  /// ``n.x + d = 0`` by the So3 element and scales it by the scale factor:
  ///
  /// Normal vector ``n`` is rotated
  /// Offset ``d`` is scaled
  ///
  SOPHUS_FUNC Hyperplane operator*(Hyperplane const& p) const {
    auto const this_scale = scale();
    return Hyperplane(
        (*this) * p.normal() / this_scale, this_scale * p.offset());
  }

  /// In-place group multiplication. This method is only valid if the return
  /// type of the multiplication is compatible with this So3's Scalar type.
  ///
  /// Note: This function performs saturation for products close to zero in
  /// order to ensure the class invariant.
  ///
  template <
      typename TOtherDerived,
      typename = typename std::enable_if<
          std::is_same<Scalar, ReturnScalar<TOtherDerived>>::value>::type>
  SOPHUS_FUNC RxSo3Base<TDerived>& operator*=(
      RxSo3Base<TOtherDerived> const& other) {
    *static_cast<TDerived*>(this) = *this * other;
    return *this;
  }

  /// Returns internal parameters of RxSO(3).
  ///
  /// It returns (q.imag[0], q.imag[1], q.imag[2], q.real), with q being the
  /// quaternion.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Vector<Scalar, kNumParameters> params()
      const {
    return quaternion().coeffs();
  }

  /// Sets non-zero quaternion
  ///
  /// Precondition: ``quat`` must not be close to either zero or infinity
  SOPHUS_FUNC void setQuaternion(Eigen::Quaternion<Scalar> const& quat) {
    FARM_ASSERT(
        quat.squaredNorm() > kEpsilon<Scalar> * kEpsilon<Scalar>,
        "Scale factor must be greater-equal epsilon.");
    FARM_ASSERT(
        quat.squaredNorm() < Scalar(1.) / (kEpsilon<Scalar> * kEpsilon<Scalar>),
        "Inverse scale factor must be greater-equal epsilon.");
    static_cast<TDerived*>(this)->quaternionNonconst() = quat;
  }

  /// Accessor of quaternion.
  ///
  SOPHUS_FUNC [[nodiscard]] QuaternionType const& quaternion() const {
    return static_cast<TDerived const*>(this)->quaternion();
  }

  /// Returns rotation matrix.
  ///
  SOPHUS_FUNC [[nodiscard]] Transformation rotationMatrix() const {
    QuaternionTemporaryType norm_quad = quaternion();
    norm_quad.normalize();
    return norm_quad.toRotationMatrix();
  }

  /// Returns scale.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar scale() const {
    return quaternion().squaredNorm();
  }

  /// Setter of quaternion using rotation matrix ``R``, leaves scale as is.
  ///
  SOPHUS_FUNC void setRotationMatrix(Transformation const& r) {
    using std::sqrt;
    Scalar saved_scale = scale();
    quaternionNonconst() = r;
    quaternionNonconst().coeffs() *= sqrt(saved_scale);
  }

  /// Sets scale and leaves rotation as is.
  ///
  /// Note: This function as a significant computational cost, since it has to
  /// call the square root twice.
  ///
  SOPHUS_FUNC
  void setScale(Scalar const& scale) {
    using std::sqrt;
    quaternionNonconst().normalize();
    quaternionNonconst().coeffs() *= sqrt(scale);
  }

  /// Setter of quaternion using scaled rotation matrix ``sR``.
  ///
  /// Precondition: The 3x3 matrix must be "scaled orthogonal"
  ///               and have a positive determinant.
  ///
  SOPHUS_FUNC void setScaledRotationMatrix(Transformation const& s_r) {
    Transformation squared_s_r = s_r * s_r.transpose();
    Scalar squared_scale =
        Scalar(1. / 3.) *
        (squared_s_r(0, 0) + squared_s_r(1, 1) + squared_s_r(2, 2));
    FARM_ASSERT(
        squared_scale >= kEpsilon<Scalar> * kEpsilon<Scalar>,
        "Scale factor must be greater-equal epsilon.");
    FARM_ASSERT(
        squared_scale < Scalar(1.) / (kEpsilon<Scalar> * kEpsilon<Scalar>),
        "Inverse scale factor must be greater-equal epsilon.");
    Scalar scale = sqrt(squared_scale);
    quaternionNonconst() = s_r / scale;
    quaternionNonconst().coeffs() *= sqrt(scale);
  }

  /// Setter of SO(3) rotations, leaves scale as is.
  ///
  SOPHUS_FUNC void setSO3(So3<Scalar> const& so3) {
    using std::sqrt;
    Scalar saved_scale = scale();
    quaternionNonconst() = so3.unitQuaternion();
    quaternionNonconst().coeffs() *= sqrt(saved_scale);
  }

  SOPHUS_FUNC [[nodiscard]] So3<Scalar> so3() const {
    return So3<Scalar>(quaternion());
  }

  /// Returns derivative of  this * RxSo3::exp(x) wrt. x at x=0
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kNumParameters, kDoF>
  dxThisMulExpXAt0() const {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> j;
    Eigen::Quaternion<Scalar> const q = quaternion();
    j.col(3) = q.coeffs() * Scalar(0.5);
    Scalar const c0 = Scalar(0.5) * q.w();
    Scalar const c1 = Scalar(0.5) * q.z();
    Scalar const c2 = -c1;
    Scalar const c3 = Scalar(0.5) * q.y();
    Scalar const c4 = Scalar(0.5) * q.x();
    Scalar const c5 = -c4;
    Scalar const c6 = -c3;
    j(0, 0) = c0;
    j(0, 1) = c2;
    j(0, 2) = c3;
    j(1, 0) = c1;
    j(1, 1) = c0;
    j(1, 2) = c5;
    j(2, 0) = c6;
    j(2, 1) = c4;
    j(2, 2) = c0;
    j(3, 0) = c5;
    j(3, 1) = c6;
    j(3, 2) = c2;
    return j;
  }

  /// Returns derivative of log(this^{-1} * x) by x at x=this.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kDoF, kNumParameters>
  dxLogThisInvTimesXAtThis() const {
    auto& q = quaternion();
    Eigen::Matrix<Scalar, kDoF, kNumParameters> j;
    // clang-format off
    j << q.w(),  q.z(), -q.y(), -q.x(),
        -q.z(),  q.w(),  q.x(), -q.y(),
         q.y(), -q.x(),  q.w(), -q.z(),
         q.x(),  q.y(),  q.z(),  q.w();
    // clang-format on
    const Scalar scaler = Scalar(2.) / q.squaredNorm();
    return j * scaler;
  }

 private:
  /// Mutator of quaternion is private to ensure class invariant.
  ///
  SOPHUS_FUNC QuaternionType& quaternionNonconst() {
    return static_cast<TDerived*>(this)->quaternionNonconst();
  }
};

/// RxSo3 using storage; derived from RxSo3Base.
template <class TScalar, int kOptions>
class RxSo3 : public RxSo3Base<RxSo3<TScalar, kOptions>> {
 public:
  using Base = RxSo3Base<RxSo3<TScalar, kOptions>>;
  static int constexpr kDoF = Base::kDoF;
  static int constexpr kNumParameters = Base::kNumParameters;

  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using QuaternionMember = Eigen::Quaternion<Scalar, kOptions>;

  /// ``Base`` is friend so quaternion_nonconst can be accessed from ``Base``.
  friend class RxSo3Base<RxSo3<TScalar, kOptions>>;

  struct RxSo3AndTheta {
    RxSo3<Scalar> rxso3;
    Scalar theta;
  };

  using Base::operator=;

  /// Define copy-assignment operator explicitly. The definition of
  /// implicit copy assignment operator is deprecated in presence of a
  /// user-declared copy constructor (-Wdeprecated-copy in clang >= 13).
  SOPHUS_FUNC RxSo3& operator=(RxSo3 const& other) = default;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Default constructor initializes quaternion to identity rotation and scale
  /// to 1.
  ///
  SOPHUS_FUNC RxSo3()
      : quaternion_(Scalar(1), Scalar(0), Scalar(0), Scalar(0)) {}

  /// Copy constructor
  ///
  SOPHUS_FUNC RxSo3(RxSo3 const& other) = default;

  /// Copy-like constructor from OtherDerived
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC RxSo3(RxSo3Base<TOtherDerived> const& other)
      : quaternion_(other.quaternion()) {}

  /// Constructor from scaled rotation matrix
  ///
  /// Precondition: rotation matrix need to be scaled orthogonal with
  /// determinant of ``s^3``.
  ///
  SOPHUS_FUNC explicit RxSo3(Transformation const& s_r) {
    this->setScaledRotationMatrix(s_r);
  }

  /// Constructor from scale factor and rotation matrix ``R``.
  ///
  /// Precondition: Rotation matrix ``R`` must to be orthogonal with determinant
  ///               of 1 and ``scale`` must not be close to either zero or
  ///               infinity.
  ///
  SOPHUS_FUNC RxSo3(Scalar const& scale, Transformation const& r)
      : quaternion_(r) {
    FARM_ASSERT(
        scale >= kEpsilon<Scalar>,
        "Scale factor must be greater-equal epsilon.");
    FARM_ASSERT(
        scale < Scalar(1.) / kEpsilon<Scalar>,
        "Inverse scale factor must be greater-equal epsilon.");
    using std::sqrt;
    quaternion_.coeffs() *= sqrt(scale);
  }

  /// Constructor from scale factor and So3
  ///
  /// Precondition: ``scale`` must not to be close to either zero or infinity.
  ///
  SOPHUS_FUNC RxSo3(Scalar const& scale, So3<Scalar> const& so3)
      : quaternion_(so3.unitQuaternion()) {
    FARM_ASSERT(
        scale >= kEpsilon<Scalar>,
        "Scale factor must be greater-equal epsilon.");
    FARM_ASSERT(
        scale < Scalar(1.) / kEpsilon<Scalar>,
        "Inverse scale factor must be greater-equal epsilon.");
    using std::sqrt;
    quaternion_.coeffs() *= sqrt(scale);
  }

  /// Constructor from quaternion
  ///
  /// Precondition: quaternion must not be close to either zero or infinity.
  ///
  template <class TD>
  SOPHUS_FUNC explicit RxSo3(Eigen::QuaternionBase<TD> const& quat)
      : quaternion_(quat) {
    static_assert(
        std::is_same<typename TD::Scalar, Scalar>::value,
        "must be same Scalar type.");
    FARM_ASSERT(
        quaternion_.squaredNorm() >= kEpsilon<Scalar>,
        "Scale factor must be greater-equal epsilon.");
    FARM_ASSERT(
        quat.squaredNorm() < Scalar(1.) / kEpsilon<Scalar>,
        "Inverse scale factor must be greater-equal epsilon.");
  }

  /// Constructor from scale factor and unit quaternion
  ///
  /// Precondition: quaternion must not be close to zero.
  ///
  template <class TD>
  SOPHUS_FUNC explicit RxSo3(
      Scalar const& scale, Eigen::QuaternionBase<TD> const& unit_quat)
      : RxSo3(scale, So3<Scalar>(unit_quat)) {}

  /// Accessor of quaternion.
  ///
  SOPHUS_FUNC [[nodiscard]] QuaternionMember const& quaternion() const {
    return quaternion_;
  }

  /// Returns derivative of exp(x) wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpXAt0() {
    static Scalar const kH(0.5);
    return kH * Eigen::Matrix<Scalar, kNumParameters, kDoF>::Identity();
  }

  /// Returns derivative of exp(x) wrt. x.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpX(
      Tangent const& a) {
    using std::exp;
    using std::sqrt;
    Eigen::Matrix<Scalar, kNumParameters, kDoF> j;
    Eigen::Vector3<Scalar> const omega = a.template head<3>();
    Scalar const sigma = a[3];
    Eigen::Quaternion<Scalar> quat = So3<Scalar>::exp(omega).unitQuaternion();
    Scalar const scale = sqrt(exp(sigma));
    Scalar const scale_half = scale * Scalar(0.5);

    j.template block<4, 3>(0, 0) = So3<Scalar>::dxExpX(omega) * scale;
    j.col(3) = quat.coeffs() * scale_half;
    return j;
  }

  /// Returns derivative of exp(x) * p wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, 3, kDoF> dxExpXTimesPointAt0(
      Point const& point) {
    Eigen::Matrix<Scalar, 3, kDoF> j;
    j << sophus::So3<Scalar>::hat(-point), point;
    return j;
  }

  /// Returns derivative of ``expmat(x)`` wrt. ``x_i at x=0``, with
  /// ``expmat(.)`` being the matrix exponential.
  ///
  SOPHUS_FUNC static Transformation dxiExpmatXAt0(int i) {
    return generator(i);
  }
  /// Group exponential
  ///
  /// This functions takes in an element of tangent space (= rotation 3-vector
  /// plus logarithm of scale) and returns the corresponding element of the
  /// group RxSo3.
  ///
  /// To be more specific, thixs function computes ``expmat(hat(omega))``
  /// with ``expmat(.)`` being the matrix exponential and ``hat(.)`` being the
  /// hat()-operator of RSO3.
  ///
  SOPHUS_FUNC static RxSo3<Scalar> exp(Tangent const& a) {
    return expAndTheta(a).rxso3;
  }

  /// As above, but also returns ``theta = |omega|``.
  ///
  SOPHUS_FUNC static RxSo3AndTheta expAndTheta(Tangent const& vec_a) {
    using std::exp;
    using std::max;
    using std::min;
    using std::sqrt;

    RxSo3AndTheta rxso3_and_theta;

    Eigen::Vector3<Scalar> const vec_omega = vec_a.template head<3>();
    Scalar sigma = vec_a[3];
    Scalar scale = exp(sigma);
    // Ensure that scale-factor contraint is always valid
    scale = max(scale, kEpsilonPlus<Scalar>);
    scale = min(scale, Scalar(1.) / kEpsilonPlus<Scalar>);
    Scalar sqrt_scale = sqrt(scale);
    auto so3_and_theta = So3<Scalar>::expAndTheta(vec_omega);
    Eigen::Quaternion<Scalar> quat = so3_and_theta.so3.unitQuaternion();
    quat.coeffs() *= sqrt_scale;
    rxso3_and_theta.rxso3 = RxSo3<Scalar>(quat);
    rxso3_and_theta.theta = so3_and_theta.theta;
    return rxso3_and_theta;
  }

  /// Returns the ith infinitesimal generators of ``R+ x SO(3)``.
  ///
  /// The infinitesimal generators of RxSo3 are:
  ///
  /// ```
  ///         |  0  0  0 |
  ///   G_0 = |  0  0 -1 |
  ///         |  0  1  0 |
  ///
  ///         |  0  0  1 |
  ///   G_1 = |  0  0  0 |
  ///         | -1  0  0 |
  ///
  ///         |  0 -1  0 |
  ///   G_2 = |  1  0  0 |
  ///         |  0  0  0 |
  ///
  ///         |  1  0  0 |
  ///   G_2 = |  0  1  0 |
  ///         |  0  0  1 |
  /// ```
  ///
  /// Precondition: ``i`` must be 0, 1, 2 or 3.
  ///
  SOPHUS_FUNC static Transformation generator(int i) {
    FARM_ASSERT(i >= 0 && i <= 3, "i should be in range [0,3].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  /// hat-operator
  ///
  /// It takes in the 4-vector representation ``a`` (= rotation vector plus
  /// logarithm of scale) and  returns the corresponding matrix representation
  /// of Lie algebra element.
  ///
  /// Formally, the hat()-operator of RxSo3 is defined as
  ///
  ///   ``hat(.): R^4 -> R^{3x3},  hat(a) = sum_i a_i * G_i``  (for i=0,1,2,3)
  ///
  /// with ``G_i`` being the ith infinitesimal generator of RxSo3.
  ///
  /// The corresponding inverse is the vee()-operator, see below.
  ///
  SOPHUS_FUNC static Transformation hat(Tangent const& a) {
    Transformation mat_a;
    // clang-format off
    mat_a <<  a(3), -a(2),  a(1),
              a(2),  a(3), -a(0),
             -a(1),  a(0),  a(3);
    // clang-format on
    return mat_a;
  }

  /// Lie bracket
  ///
  /// It computes the Lie bracket of RxSO(3). To be more specific, it computes
  ///
  ///   ``[omega_1, omega_2]_rxso3 := vee([hat(omega_1), hat(omega_2)])``
  ///
  /// with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.)`` the
  /// hat()-operator and ``vee(.)`` the vee()-operator of RxSo3.
  ///
  SOPHUS_FUNC static Tangent lieBracket(Tangent const& a, Tangent const& b) {
    Eigen::Vector3<Scalar> const omega1 = a.template head<3>();
    Eigen::Vector3<Scalar> const omega2 = b.template head<3>();
    Eigen::Vector4<Scalar> res;
    res.template head<3>() = omega1.cross(omega2);
    res[3] = Scalar(0);
    return res;
  }

  /// Draw uniform sample from RxSO(3) manifold.
  ///
  /// The scale factor is drawn uniformly in log2-space from [-1, 1],
  /// hence the scale is in [0.5, 2].
  ///
  template <class TUniformRandomBitGenerator>
  static RxSo3 sampleUniform(TUniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    using std::exp2;
    return RxSo3(
        exp2(uniform(generator)), So3<Scalar>::sampleUniform(generator));
  }

  /// vee-operator
  ///
  /// It takes the 3x3-matrix representation ``Omega`` and maps it to the
  /// corresponding vector representation of Lie algebra.
  ///
  /// This is the inverse of the hat()-operator, see above.
  ///
  /// Precondition: ``Omega`` must have the following structure:
  ///
  ///                |  d -c  b |
  ///                |  c  d -a |
  ///                | -b  a  d |
  ///
  SOPHUS_FUNC static Tangent vee(Transformation const& omega) {
    using std::abs;
    return Tangent(omega(2, 1), omega(0, 2), omega(1, 0), omega(0, 0));
  }

 protected:
  SOPHUS_FUNC QuaternionMember& quaternionNonconst() { return quaternion_; }

  QuaternionMember quaternion_;  // NOLINT
};

}  // namespace sophus

namespace Eigen {  // NOLINT

/// Specialization of Eigen::Map for ``RxSo3``; derived from RxSo3Base
///
/// Allows us to wrap RxSo3 objects around POD array (e.g. external c style
/// quaternion).
template <class TScalar, int kOptions>
class Map<sophus::RxSo3<TScalar>, kOptions>
    : public sophus::RxSo3Base<Map<sophus::RxSo3<TScalar>, kOptions>> {
 public:
  using Base = sophus::RxSo3Base<Map<sophus::RxSo3<TScalar>, kOptions>>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  /// ``Base`` is friend so quaternion_nonconst can be accessed from ``Base``.
  friend class sophus::RxSo3Base<Map<sophus::RxSo3<TScalar>, kOptions>>;

  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar* coeffs) : quaternion_(coeffs) {}

  /// Accessor of quaternion.
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Quaternion<Scalar>, kOptions> const&
  quaternion() const {
    return quaternion_;
  }

 protected:
  SOPHUS_FUNC Map<Eigen::Quaternion<Scalar>, kOptions>& quaternionNonconst() {
    return quaternion_;
  }

  Map<Eigen::Quaternion<Scalar>, kOptions> quaternion_;  // NOLINT
};

/// Specialization of Eigen::Map for ``RxSo3 const``; derived from RxSo3Base.
///
/// Allows us to wrap RxSo3 objects around POD array (e.g. external c style
/// quaternion).
template <class TScalar, int kOptions>
class Map<sophus::RxSo3<TScalar> const, kOptions>
    : public sophus::RxSo3Base<Map<sophus::RxSo3<TScalar> const, kOptions>> {
 public:
  using Base = sophus::RxSo3Base<Map<sophus::RxSo3<TScalar> const, kOptions>>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  explicit Map(Scalar const* coeffs) : quaternion_(coeffs) {}

  /// Accessor of quaternion.
  ///
  SOPHUS_FUNC
  [[nodiscard]] Map<Eigen::Quaternion<Scalar> const, kOptions> const&
  quaternion() const {
    return quaternion_;
  }

 protected:
  Map<Eigen::Quaternion<Scalar> const, kOptions> quaternion_;  // NOLINT
};
}  // namespace Eigen
