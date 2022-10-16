// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Special orthogonal group SO(3) - rotation in 3d.

#pragma once

#include "sophus/common/types.h"
#include "sophus/lie/so2.h"
#include "sophus/linalg/rotation_matrix.h"

// Include only the selective set of Eigen headers that we need.
// This helps when using Sophus with unusual compilers, like nvcc.
#include <Eigen/src/Geometry/OrthoMethods.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/src/Geometry/RotationBase.h>

#include <optional>

namespace sophus {
template <class TScalar, int kOptions = 0>
class So3;
using So3F64 = So3<double>;
using So3F32 = So3<float>;

template <class TScalar, int kOptions = 0>
/* [[deprecated]] */ using SO3 = So3<TScalar, kOptions>;
/* [[deprecated]] */ using SO3d = So3F64;
/* [[deprecated]] */ using SO3f = So3F32;
}  // namespace sophus

namespace Eigen {  // NOLINT
namespace internal {

template <class TScalar, int kOptionsT>
struct traits<sophus::So3<TScalar, kOptionsT>> {
  static int constexpr kOptions = kOptionsT;
  using Scalar = TScalar;
  using QuaternionType = Eigen::Quaternion<Scalar, kOptions>;
};

template <class TScalar, int kOptionsT>
struct traits<Map<sophus::So3<TScalar>, kOptionsT>>
    : traits<sophus::So3<TScalar, kOptionsT>> {
  static int constexpr kOptions = kOptionsT;
  using Scalar = TScalar;
  using QuaternionType = Map<Eigen::Quaternion<Scalar>, kOptions>;
};

template <class TScalar, int kOptionsT>
struct traits<Map<sophus::So3<TScalar> const, kOptionsT>>
    : traits<sophus::So3<TScalar, kOptionsT> const> {
  static int constexpr kOptions = kOptionsT;
  using Scalar = TScalar;
  using QuaternionType = Map<Eigen::Quaternion<Scalar> const, kOptions>;
};
}  // namespace internal
}  // namespace Eigen

namespace sophus {

/// So3 base type - implements So3 class but is storage agnostic.
///
/// SO(3) is the group of rotations in 3d. As a matrix group, it is the set of
/// matrices which are orthogonal such that ``R * R' = I`` (with ``R'`` being
/// the transpose of ``R``) and have a positive determinant. In particular, the
/// determinant is 1. Internally, the group is represented as a unit quaternion.
/// Unit quaternion can be seen as members of the special unitary group SU(2).
/// SU(2) is a double cover of SO(3). Hence, for every rotation matrix ``R``,
/// there exist two unit quaternions: ``(r, v)`` and ``(-r, -v)``, with ``r``
/// the real part and ``v`` being the imaginary 3-vector part of the quaternion.
///
/// SO(3) is a compact, but non-commutative group. First it is compact since the
/// set of rotation matrices is a closed and bounded set. Second it is
/// non-commutative since the equation ``R_1 * R_2 = R_2 * R_1`` does not hold
/// in general. For example rotating an object by some degrees about its
/// ``x``-axis and then by some degrees about its y axis, does not lead to the
/// same orientation when rotation first about ``y`` and then about ``x``.
///
/// Class invariant: The 2-norm of ``unit_quaternion`` must be close to 1.
/// Technically speaking, it must hold that:
///
///   ``|unit_quaternion().squaredNorm() - 1| <= Constants::epsilon()``.
template <class TDerived>
class So3Base {
 public:
  static int constexpr kOptions = Eigen::internal::traits<TDerived>::kOptions;
  using Scalar = typename Eigen::internal::traits<TDerived>::Scalar;
  using QuaternionType =
      typename Eigen::internal::traits<TDerived>::QuaternionType;
  using QuaternionTemporaryType = Eigen::Quaternion<Scalar, kOptions>;

  /// Degrees of freedom of group, number of dimensions in tangent space.
  static int constexpr kDoF = 3;
  /// Number of internal parameters used (quaternion is a 4-tuple).
  static int constexpr kNumParameters = 4;
  /// Group transformations are 3x3 matrices.
  static int constexpr kMatrixDim = 3;
  /// Points are 3-dimensional/ Include only the selective set of Eigen headers
  /// that we need.
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
  /// double scalars with So3 operations.
  template <class TOtherDerived>
  using ReturnScalar = typename Eigen::
      ScalarBinaryOpTraits<Scalar, typename TOtherDerived::Scalar>::ReturnType;

  template <class TOtherDerived>
  using So3Product = So3<ReturnScalar<TOtherDerived>>;

  template <class TPointDerived>
  using PointProduct = Eigen::Vector3<ReturnScalar<TPointDerived>>;

  template <class THPointDerived>
  using HomogeneousPointProduct = Eigen::Vector4<ReturnScalar<THPointDerived>>;

  /// Adjoint transformation
  //
  /// This function return the adjoint transformation ``Ad`` of the group
  /// element ``A`` such that for all ``x`` it holds that
  /// ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  //
  /// For SO(3), it simply returns the rotation matrix corresponding to ``A``.
  ///
  SOPHUS_FUNC [[nodiscard]] Adjoint adj() const { return matrix(); }

  /// Extract rotation angle about canonical X-axis
  ///
  template <class TS = Scalar>
  SOPHUS_FUNC
      [[nodiscard]] std::enable_if_t<std::is_floating_point<TS>::value, TS>
      angleX() const {
    Eigen::Matrix3<Scalar> r = matrix();
    Eigen::Matrix2<Scalar> rx = r.template block<2, 2>(1, 1);
    return So2<Scalar>(makeRotationMatrix(rx)).log();
  }

  /// Extract rotation angle about canonical Y-axis
  ///
  template <class TS = Scalar>
  SOPHUS_FUNC
      [[nodiscard]] std::enable_if_t<std::is_floating_point<TS>::value, TS>
      angleY() const {
    Eigen::Matrix3<Scalar> r = matrix();
    Eigen::Matrix2<Scalar> ry;
    // clang-format off
    ry <<
      r(0, 0), r(2, 0),
      r(0, 2), r(2, 2);
    // clang-format on
    return So2<Scalar>(makeRotationMatrix(ry)).log();
  }

  /// Extract rotation angle about canonical Z-axis
  ///
  template <class TS = Scalar>
  SOPHUS_FUNC
      [[nodiscard]] std::enable_if_t<std::is_floating_point<TS>::value, TS>
      angleZ() const {
    Eigen::Matrix3<Scalar> r = matrix();
    Eigen::Matrix2<Scalar> rz = r.template block<2, 2>(0, 0);
    return So2<Scalar>(makeRotationMatrix(rz)).log();
  }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class TNewScalarType>
  SOPHUS_FUNC [[nodiscard]] So3<TNewScalarType> cast() const {
    return So3<TNewScalarType>(
        unitQuaternion().template cast<TNewScalarType>());
  }

  /// This provides unsafe read/write access to internal data. SO(3) is
  /// represented by an Eigen::Quaternion (four parameters). When using direct
  /// write access, the user needs to take care of that the quaternion stays
  /// normalized.
  ///
  /// Note: The first three Scalars represent the imaginary parts, while the
  /// forth Scalar represent the real part.
  ///
  SOPHUS_FUNC Scalar* data() { return mutUnitQuaternion().coeffs().data(); }

  /// Const version of data() above.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar const* data() const {
    return unitQuaternion().coeffs().data();
  }

  /// Returns derivative of  this * So3::exp(x)  wrt. x at x=0.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kNumParameters, kDoF>
  dxThisMulExpXAt0() const {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> j;
    Eigen::Quaternion<Scalar> const q = unitQuaternion();
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

  /// Returns derivative of log(this^{-1} * x) wrt. x at x=this.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kDoF, kNumParameters>
  dxLogThisInvTimesXAtThis() const {
    auto& q = unitQuaternion();
    Eigen::Matrix<Scalar, kDoF, kNumParameters> j;
    // clang-format off
    j << q.w(),  q.z(), -q.y(), -q.x(),
        -q.z(),  q.w(),  q.x(), -q.y(),
         q.y(), -q.x(),  q.w(), -q.z();
    // clang-format on
    return j * Scalar(2.);
  }

  /// Returns internal parameters of SO(3).
  ///
  /// It returns (q.imag[0], q.imag[1], q.imag[2], q.real), with q being the
  /// unit quaternion.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Vector<Scalar, kNumParameters> params()
      const {
    return unitQuaternion().coeffs();
  }

  /// Returns group inverse.
  ///
  SOPHUS_FUNC [[nodiscard]] So3<Scalar> inverse() const {
    return So3<Scalar>(unitQuaternion().conjugate());
  }

  /// Logarithmic map
  ///
  /// Computes the logarithm, the inverse of the group exponential which maps
  /// element of the group (rotation matrices) to elements of the tangent space
  /// (rotation-vector).
  ///
  /// To be specific, this function computes ``vee(logmat(.))`` with
  /// ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  /// of SO(3).
  ///
  SOPHUS_FUNC [[nodiscard]] Tangent log() const {
    return logAndTheta().tangent;
  }

  /// As above, but also returns ``theta = |omega|``.
  ///
  SOPHUS_FUNC [[nodiscard]] TangentAndTheta logAndTheta() const {
    TangentAndTheta j;
    using std::abs;
    using std::atan2;
    using std::sqrt;
    Scalar squared_n = unitQuaternion().vec().squaredNorm();
    Scalar w = unitQuaternion().w();

    Scalar two_atan_nbyw_by_n;

    /// Atan-based log thanks to
    ///
    /// C. Hertzberg et al.:
    /// "Integrating Generic Sensor Fusion Algorithms with Sound State
    /// Representation through Encapsulation of Manifolds"
    /// Information Fusion, 2011

    if (squared_n < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      // If quaternion is normalized and n=0, then w should be 1;
      // w=0 should never happen here!
      FARM_CHECK(
          abs(w) >= kEpsilon<Scalar>,
          "Quaternion ({}) should be normalized!",
          unitQuaternion().coeffs().transpose().eval());
      Scalar squared_w = w * w;
      two_atan_nbyw_by_n =
          Scalar(2) / w - Scalar(2.0 / 3.0) * (squared_n) / (w * squared_w);
      j.theta = Scalar(2) * squared_n / w;
    } else {
      Scalar n = sqrt(squared_n);

      // w < 0 ==> cos(theta/2) < 0 ==> theta > pi
      //
      // By convention, the condition |theta| < pi is imposed by wrapping theta
      // to pi; The wrap operation can be folded inside evaluation of atan2
      //
      // theta - pi = atan(sin(theta - pi), cos(theta - pi))
      //            = atan(-sin(theta), -cos(theta))
      //
      Scalar atan_nbyw =
          (w < Scalar(0)) ? Scalar(atan2(-n, -w)) : Scalar(atan2(n, w));
      two_atan_nbyw_by_n = Scalar(2) * atan_nbyw / n;
      j.theta = two_atan_nbyw_by_n * n;
    }

    j.tangent = two_atan_nbyw_by_n * unitQuaternion().vec();
    return j;
  }

  /// It re-normalizes ``unit_quaternion`` to unit length.
  ///
  /// Note: Because of the class invariant, there is typically no need to call
  /// this function directly.
  ///
  SOPHUS_FUNC void normalize() {
    Scalar length = mutUnitQuaternion().norm();
    FARM_CHECK(
        length >= kEpsilon<Scalar>,
        "Quaternion ({}) should not be close to zero!",
        mutUnitQuaternion().coeffs().transpose());
    mutUnitQuaternion().coeffs() /= length;
  }

  /// Returns 3x3 matrix representation of the instance.
  ///
  /// For SO(3), the matrix representation is an orthogonal matrix ``R`` with
  /// ``det(R)=1``, thus the so-called "rotation matrix".
  ///
  SOPHUS_FUNC [[nodiscard]] Transformation matrix() const {
    return unitQuaternion().toRotationMatrix();
  }

  /// Assignment-like operator from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC So3Base<TDerived>& operator=(
      So3Base<TOtherDerived> const& other) {
    mutUnitQuaternion() = other.unitQuaternion();
    return *this;
  }

  template <
      typename TQuaternionProductType,
      typename TQuaternionTypeA,
      typename TQuaternionTypeB>
  static TQuaternionProductType quaternionProduct(
      TQuaternionTypeA const& a, TQuaternionTypeB const& b) {
    return TQuaternionProductType(
        a.w() * b.w() - a.x() * b.x() - a.y() * b.y() - a.z() * b.z(),
        a.w() * b.x() + a.x() * b.w() + a.y() * b.z() - a.z() * b.y(),
        a.w() * b.y() + a.y() * b.w() + a.z() * b.x() - a.x() * b.z(),
        a.w() * b.z() + a.z() * b.w() + a.x() * b.y() - a.y() * b.x());
  }

  /// Group multiplication, which is rotation concatenation.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC So3Product<TOtherDerived> operator*(
      So3Base<TOtherDerived> const& other) const {
    using QuaternionProductType =
        typename So3Product<TOtherDerived>::QuaternionType;
    QuaternionType const& a = unitQuaternion();
    const typename TOtherDerived::QuaternionType& b = other.unitQuaternion();
    /// NOTE: We cannot use Eigen's Quaternion multiplication because it always
    /// returns a Quaternion of the same Scalar as this object, so it is not
    /// able to multiple Jets and doubles correctly.
    return So3Product<TOtherDerived>(
        quaternionProduct<QuaternionProductType>(a, b));
  }

  /// Group action on 3-points.
  ///
  /// This function rotates a 3 dimensional point ``p`` by the So3 element
  ///  ``bar_R_foo`` (= rotation matrix): ``p_bar = bar_R_foo * p_foo``.
  ///
  /// Since So3 is internally represented by a unit quaternion ``q``, it is
  /// implemented as ``p_bar = q * p_foo * q^{*}``
  /// with ``q^{*}`` being the quaternion conjugate of ``q``.
  ///
  /// Geometrically, ``p``  is rotated by angle ``|omega|`` around the
  /// axis ``omega/|omega|`` with ``omega := vee(log(bar_R_foo))``.
  ///
  /// For ``vee``-operator, see below.
  ///
  template <
      typename TPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<TPointDerived, 3>::value>::type>
  SOPHUS_FUNC PointProduct<TPointDerived> operator*(
      Eigen::MatrixBase<TPointDerived> const& p) const {
    /// NOTE: We cannot use Eigen's Quaternion transformVector because it always
    /// returns aEigen::Vector3 of the same Scalar as this quaternion, so it is
    /// not able to be applied to Jets and doubles correctly.
    QuaternionType const& q = unitQuaternion();
    PointProduct<TPointDerived> uv = q.vec().cross(p);
    uv += uv;
    return p + q.w() * uv + q.vec().cross(uv);
  }

  /// Group action on homogeneous 3-points. See above for more details.
  template <
      typename THPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<THPointDerived, 4>::value>::type>
  SOPHUS_FUNC HomogeneousPointProduct<THPointDerived> operator*(
      Eigen::MatrixBase<THPointDerived> const& p) const {
    auto const rp = *this * p.template head<3>();
    return HomogeneousPointProduct<THPointDerived>(rp(0), rp(1), rp(2), p(3));
  }

  /// Group action on lines.
  ///
  /// This function rotates a parametrized line ``l(t) = o + t * d`` by the So3
  /// element:
  ///
  /// Both direction ``d`` and origin ``o`` are rotated as a 3 dimensional point
  ///
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), (*this) * l.direction());
  }

  /// Group action on planes.
  ///
  /// This function rotates a plane
  /// ``n.x + d = 0`` by the So3 element:
  ///
  /// Normal vector ``n`` is rotated
  /// Offset ``d`` is left unchanged
  ///
  SOPHUS_FUNC Hyperplane operator*(Hyperplane const& p) const {
    return Hyperplane((*this) * p.normal(), p.offset());
  }

  /// In-place group multiplication. This method is only valid if the return
  /// type of the multiplication is compatible with this So3's Scalar type.
  ///
  template <
      typename TOtherDerived,
      typename = typename std::enable_if<
          std::is_same<Scalar, ReturnScalar<TOtherDerived>>::value>::type>
  SOPHUS_FUNC So3Base<TDerived>& operator*=(
      So3Base<TOtherDerived> const& other) {
    *static_cast<TDerived*>(this) = *this * other;
    return *this;
  }

  /// Takes in quaternion, and normalizes it.
  ///
  /// Precondition: The quaternion must not be close to zero.
  ///
  SOPHUS_FUNC void setQuaternion(Eigen::Quaternion<Scalar> const& quaternion) {
    mutUnitQuaternion() = quaternion;
    normalize();
  }

  /// Accessor of unit quaternion.
  ///
  SOPHUS_FUNC [[nodiscard]] QuaternionType const& unitQuaternion() const {
    return static_cast<TDerived const*>(this)->unitQuaternion();
  }

 private:
  /// Mutator of unit_quaternion is private to ensure class invariant. That is
  /// the quaternion must stay close to unit length.;
  ///
  SOPHUS_FUNC QuaternionType& mutUnitQuaternion() {
    return static_cast<TDerived*>(this)->mutUnitQuaternion();
  }
};

/// So3 using default storage; derived from So3Base.
template <class TScalar, int kOptions>
class So3 : public So3Base<So3<TScalar, kOptions>> {
 public:
  using Base = So3Base<So3<TScalar, kOptions>>;
  static int constexpr kDoF = Base::kDoF;
  static int constexpr kNumParameters = Base::kNumParameters;

  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using QuaternionMember = Eigen::Quaternion<Scalar, kOptions>;

  struct So3AndTheta {
    So3<Scalar> so3;
    Scalar theta;
  };

  /// ``Base`` is friend so unit_quaternion_nonconst can be accessed from
  /// ``Base``.
  friend class So3Base<So3<Scalar, kOptions>>;

  using Base::operator=;

  /// Define copy-assignment operator explicitly. The definition of
  /// implicit copy assignment operator is deprecated in presence of a
  /// user-declared copy constructor (-Wdeprecated-copy in clang >= 13).
  SOPHUS_FUNC So3& operator=(So3 const& other) = default;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Default constructor initializes unit quaternion to identity rotation.
  ///
  SOPHUS_FUNC So3()
      : unit_quaternion_(Scalar(1), Scalar(0), Scalar(0), Scalar(0)) {}

  /// Copy constructor
  ///
  SOPHUS_FUNC So3(So3 const& other) = default;

  /// Copy-like constructor from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC So3(So3Base<TOtherDerived> const& other)
      : unit_quaternion_(other.unitQuaternion()) {}

  /// Constructor from rotation matrix
  ///
  /// Precondition: rotation matrix needs to be orthogonal with determinant
  /// of 1.
  ///
  SOPHUS_FUNC So3(Transformation const& r) : unit_quaternion_(r) {
    FARM_CHECK(isOrthogonal(r), "R is not orthogonal:\n {}", r * r.transpose());
    FARM_CHECK(
        r.determinant() > Scalar(0),
        "det(R) is not positive: {}",
        r.determinant());
  }

  /// Constructor from quaternion
  ///
  /// Precondition: quaternion must not be close to zero.
  ///
  template <class TD>
  SOPHUS_FUNC explicit So3(Eigen::QuaternionBase<TD> const& quat)
      : unit_quaternion_(quat) {
    static_assert(
        std::is_same<typename Eigen::QuaternionBase<TD>::Scalar, Scalar>::value,
        "Input must be of same scalar type");
    Base::normalize();
  }

  /// Accessor of unit quaternion.
  ///
  SOPHUS_FUNC [[nodiscard]] QuaternionMember const& unitQuaternion() const {
    return unit_quaternion_;
  }

  /// Returns the left Jacobian on lie group. See 1st entry in rightmost column
  /// in: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17_identities.pdf
  ///
  /// A precomputed `theta` can be optionally passed in
  ///
  /// Warning: Not to be confused with dxExpX(), which is derivative of the
  ///          internal quaternion representation of So3 wrt the tangent vector
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kDoF, kDoF> leftJacobian(
      Tangent const& omega,
      std::optional<Scalar> const& maybe_theta = std::nullopt) {
    using std::cos;
    using std::sin;
    using std::sqrt;

    Scalar const theta_sq =
        maybe_theta ? *maybe_theta * *maybe_theta : omega.squaredNorm();
    Eigen::Matrix3<Scalar> const mat_omega = So3<Scalar>::hat(omega);
    Eigen::Matrix3<Scalar> const mat_omega_sq = mat_omega * mat_omega;
    Eigen::Matrix3<Scalar> v;

    if (theta_sq < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      v = Eigen::Matrix3<Scalar>::Identity() + Scalar(0.5) * mat_omega;
    } else {
      Scalar theta = maybe_theta ? *maybe_theta : sqrt(theta_sq);
      v = Eigen::Matrix3<Scalar>::Identity() +
          (Scalar(1) - cos(theta)) / theta_sq * mat_omega +
          (theta - sin(theta)) / (theta_sq * theta) * mat_omega_sq;
    }
    return v;
  }

  SOPHUS_FUNC static Eigen::Matrix<Scalar, kDoF, kDoF> leftJacobianInverse(
      Tangent const& omega,
      std::optional<Scalar> const& maybe_theta = std::nullopt) {
    using std::cos;
    using std::sin;
    using std::sqrt;
    Scalar const theta_sq =
        maybe_theta ? *maybe_theta * *maybe_theta : omega.squaredNorm();
    Eigen::Matrix3<Scalar> const mat_omega = So3<Scalar>::hat(omega);

    Eigen::Matrix3<Scalar> v_inv;
    if (theta_sq < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      v_inv = Eigen::Matrix3<Scalar>::Identity() - Scalar(0.5) * mat_omega +
              Scalar(1. / 12.) * (mat_omega * mat_omega);

    } else {
      Scalar const theta = maybe_theta ? *maybe_theta : sqrt(theta_sq);
      Scalar const half_theta = Scalar(0.5) * theta;

      v_inv = Eigen::Matrix3<Scalar>::Identity() - Scalar(0.5) * mat_omega +
              (Scalar(1) -
               Scalar(0.5) * theta * cos(half_theta) / sin(half_theta)) /
                  (theta * theta) * (mat_omega * mat_omega);
    }
    return v_inv;
  }

  /// Returns derivative of  this * x  wrt x at x=0.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kDoF, kDoF> dxThisMulX()
      const {
    return this->matrix();
  }

  /// Returns derivative of exp(x) wrt. x.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpX(
      Tangent const& omega) {
    using std::cos;
    using std::exp;
    using std::sin;
    using std::sqrt;
    Scalar const c0 = omega[0] * omega[0];
    Scalar const c1 = omega[1] * omega[1];
    Scalar const c2 = omega[2] * omega[2];
    Scalar const c3 = c0 + c1 + c2;

    if (c3 < kEpsilon<Scalar>) {
      return dxExpXAt0();
    }

    Scalar const c4 = sqrt(c3);
    Scalar const c5 = 1.0 / c4;
    Scalar const c6 = 0.5 * c4;
    Scalar const c7 = sin(c6);
    Scalar const c8 = c5 * c7;
    Scalar const c9 = pow(c3, -3.0L / 2.0L);
    Scalar const c10 = c7 * c9;
    Scalar const c11 = Scalar(1.0) / c3;
    Scalar const c12 = cos(c6);
    Scalar const c13 = Scalar(0.5) * c11 * c12;
    Scalar const c14 = c7 * c9 * omega[0];
    Scalar const c15 = Scalar(0.5) * c11 * c12 * omega[0];
    Scalar const c16 = -c14 * omega[1] + c15 * omega[1];
    Scalar const c17 = -c14 * omega[2] + c15 * omega[2];
    Scalar const c18 = omega[1] * omega[2];
    Scalar const c19 = -c10 * c18 + c13 * c18;
    Scalar const c20 = Scalar(0.5) * c5 * c7;
    Eigen::Matrix<Scalar, kNumParameters, kDoF> j;
    j(0, 0) = -c0 * c10 + c0 * c13 + c8;
    j(0, 1) = c16;
    j(0, 2) = c17;
    j(1, 0) = c16;
    j(1, 1) = -c1 * c10 + c1 * c13 + c8;
    j(1, 2) = c19;
    j(2, 0) = c17;
    j(2, 1) = c19;
    j(2, 2) = -c10 * c2 + c13 * c2 + c8;
    j(3, 0) = -c20 * omega[0];
    j(3, 1) = -c20 * omega[1];
    j(3, 2) = -c20 * omega[2];
    return j;
  }

  /// Returns derivative of exp(x) wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParameters, kDoF> dxExpXAt0() {
    Eigen::Matrix<Scalar, kNumParameters, kDoF> j;
    // clang-format off
    j <<  Scalar(0.5),   Scalar(0),   Scalar(0),
            Scalar(0), Scalar(0.5),   Scalar(0),
            Scalar(0),   Scalar(0), Scalar(0.5),
            Scalar(0),   Scalar(0),   Scalar(0);
    // clang-format on
    return j;
  }

  /// Returns derivative of exp(x) * p wrt. x at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, 3, kDoF> dxExpXTimesPointAt0(
      Point const& point) {
    return hat(-point);
  }

  /// Returns derivative of ``expmat(x)`` wrt. ``x_i at x=0``, with
  /// ``expmat(.)`` being the matrix exponential.
  ///
  SOPHUS_FUNC static Transformation dxiExpmatXAt0(int i) {
    return generator(i);
  }

  /// Group exponential
  ///
  /// This functions takes in an element of tangent space (= rotation vector
  /// ``omega``) and returns the corresponding element of the group SO(3).
  ///
  /// To be more specific, this function computes ``expmat(hat(omega))``
  /// with ``expmat(.)`` being the matrix exponential and ``hat(.)`` being the
  /// hat()-operator of SO(3).
  ///
  SOPHUS_FUNC static So3<Scalar> exp(Tangent const& omega) {
    return expAndTheta(omega).so3;
  }

  /// As above, but also returns ``theta = |omega|`` as out-parameter.
  ///
  /// Precondition: ``theta`` must not be ``nullptr``.
  ///
  SOPHUS_FUNC static So3AndTheta expAndTheta(Tangent const& omega) {
    So3AndTheta so3_and_theta;

    using std::abs;
    using std::cos;
    using std::sin;
    using std::sqrt;
    Scalar theta_sq = omega.squaredNorm();

    Scalar imag_factor;
    Scalar real_factor;
    if (theta_sq < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      so3_and_theta.theta = Scalar(0);
      Scalar theta_po4 = theta_sq * theta_sq;
      imag_factor = Scalar(0.5) - Scalar(1.0 / 48.0) * theta_sq +
                    Scalar(1.0 / 3840.0) * theta_po4;
      real_factor = Scalar(1) - Scalar(1.0 / 8.0) * theta_sq +
                    Scalar(1.0 / 384.0) * theta_po4;
    } else {
      so3_and_theta.theta = sqrt(theta_sq);
      Scalar half_theta = Scalar(0.5) * (so3_and_theta.theta);
      Scalar sin_half_theta = sin(half_theta);
      imag_factor = sin_half_theta / (so3_and_theta.theta);
      real_factor = cos(half_theta);
    }

    So3 so3;
    so3.mutUnitQuaternion() = QuaternionMember(
        real_factor,
        imag_factor * omega.x(),
        imag_factor * omega.y(),
        imag_factor * omega.z());
    so3_and_theta.so3 = so3;
    FARM_CHECK(
        abs(so3_and_theta.so3.unitQuaternion().squaredNorm() - Scalar(1)) <
            sophus::kEpsilon<Scalar>,
        "So3::exp failed! omega: {}, real: {}, img: {}",
        omega.transpose().eval(),
        real_factor,
        imag_factor);
    return so3_and_theta;
  }

  /// Returns closest So3 given arbitrary 3x3 matrix.
  ///
  template <class TS = Scalar>
  static SOPHUS_FUNC std::enable_if_t<std::is_floating_point<TS>::value, So3>
  fitToSo3(Transformation const& r) {
    return So3(::sophus::makeRotationMatrix(r));
  }

  /// Returns the ith infinitesimal generators of SO(3).
  ///
  /// The infinitesimal generators of SO(3) are:
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
  /// ```
  ///
  /// Precondition: ``i`` must be 0, 1 or 2.
  ///
  SOPHUS_FUNC static Transformation generator(int i) {
    FARM_CHECK(i >= 0 && i <= 2, "i should be in range [0,2].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  /// hat-operator
  ///
  /// It takes in the 3-vector representation ``omega`` (= rotation vector) and
  /// returns the corresponding matrix representation of Lie algebra element.
  ///
  /// Formally, the hat()-operator of SO(3) is defined as
  ///
  ///   ``hat(.): R^3 -> R^{3x3},  hat(omega) = sum_i omega_i * G_i``
  ///   (for i=0,1,2)
  ///
  /// with ``G_i`` being the ith infinitesimal generator of SO(3).
  ///
  /// The corresponding inverse is the vee()-operator, see below.
  ///
  SOPHUS_FUNC static Transformation hat(Tangent const& omega) {
    Transformation mat_omega;
    // clang-format off
    mat_omega <<
        Scalar(0), -omega(2),  omega(1),
         omega(2), Scalar(0), -omega(0),
        -omega(1),  omega(0), Scalar(0);
    // clang-format on
    return mat_omega;
  }

  /// Lie bracket
  ///
  /// It computes the Lie bracket of SO(3). To be more specific, it computes
  ///
  ///   ``[omega_1, omega_2]_so3 := vee([hat(omega_1), hat(omega_2)])``
  ///
  /// with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.)`` the
  /// hat()-operator and ``vee(.)`` the vee()-operator of So3.
  ///
  /// For the Lie algebra so3, the Lie bracket is simply the cross product:
  ///
  /// ``[omega_1, omega_2]_so3 = omega_1 x omega_2.``
  ///
  SOPHUS_FUNC static Tangent lieBracket(
      Tangent const& omega1, Tangent const& omega2) {
    return omega1.cross(omega2);
  }

  /// Construct x-axis rotation.
  ///
  static SOPHUS_FUNC So3 rotX(Scalar const& x) {
    return So3::exp(Eigen::Vector3<Scalar>(x, Scalar(0), Scalar(0)));
  }

  /// Construct y-axis rotation.
  ///
  static SOPHUS_FUNC So3 rotY(Scalar const& y) {
    return So3::exp(Eigen::Vector3<Scalar>(Scalar(0), y, Scalar(0)));
  }

  /// Construct z-axis rotation.
  ///
  static SOPHUS_FUNC So3 rotZ(Scalar const& z) {
    return So3::exp(Eigen::Vector3<Scalar>(Scalar(0), Scalar(0), z));
  }

  /// Draw uniform sample from SO(3) manifold.
  /// Based on: http://planning.cs.uiuc.edu/node198.html
  ///
  template <std::uniform_random_bit_generator TUniformRandomBitGenerator>
  static So3 sampleUniform(TUniformRandomBitGenerator& generator) {
    static_assert(
        kIsUniformRandomBitGeneratorV<TUniformRandomBitGenerator>,
        "generator must meet the UniformRandomBitGenerator concept");

    std::uniform_real_distribution<Scalar> uniform(Scalar(0), Scalar(1));
    std::uniform_real_distribution<Scalar> uniform_twopi(
        Scalar(0), 2 * kPi<Scalar>);

    const Scalar u1 = uniform(generator);
    const Scalar u2 = uniform_twopi(generator);
    const Scalar u3 = uniform_twopi(generator);

    const Scalar a = sqrt(1 - u1);
    const Scalar b = sqrt(u1);

    return So3(
        QuaternionMember(a * sin(u2), a * cos(u2), b * sin(u3), b * cos(u3)));
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
  ///                |  0 -c  b |
  ///                |  c  0 -a |
  ///                | -b  a  0 |
  ///
  SOPHUS_FUNC static Tangent vee(Transformation const& omega) {
    return Tangent(omega(2, 1), omega(0, 2), omega(1, 0));
  }

 protected:
  /// Mutator of unit_quaternion is protected to ensure class invariant.
  ///
  SOPHUS_FUNC QuaternionMember& mutUnitQuaternion() { return unit_quaternion_; }

  QuaternionMember unit_quaternion_;  // NOLINT
};

}  // namespace sophus

namespace Eigen {  // NOLINT
/// Specialization of Eigen::Map for ``So3``; derived from So3Base.
///
/// Allows us to wrap So3 objects around POD array (e.g. external c style
/// quaternion).
template <class TScalar, int kOptions>
class Map<sophus::So3<TScalar>, kOptions>
    : public sophus::So3Base<Map<sophus::So3<TScalar>, kOptions>> {
 public:
  using Base = sophus::So3Base<Map<sophus::So3<TScalar>, kOptions>>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  /// ``Base`` is friend so unit_quaternion_nonconst can be accessed from
  /// ``Base``.
  friend class sophus::So3Base<Map<sophus::So3<TScalar>, kOptions>>;

  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar* coeffs) : unit_quaternion_(coeffs) {}

  /// Accessor of unit quaternion.
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Quaternion<Scalar>, kOptions> const&
  unitQuaternion() const {
    return unit_quaternion_;
  }

 protected:
  /// Mutator of unit_quaternion is protected to ensure class invariant.
  ///
  SOPHUS_FUNC Map<Eigen::Quaternion<Scalar>, kOptions>& mutUnitQuaternion() {
    return unit_quaternion_;
  }

  Map<Eigen::Quaternion<Scalar>, kOptions> unit_quaternion_;  // NOLINT
};

/// Specialization of Eigen::Map for ``So3 const``; derived from So3Base.
///
/// Allows us to wrap So3 objects around POD array (e.g. external c style
/// quaternion).
template <class TScalar, int kOptions>
class Map<sophus::So3<TScalar> const, kOptions>
    : public sophus::So3Base<Map<sophus::So3<TScalar> const, kOptions>> {
 public:
  using Base = sophus::So3Base<Map<sophus::So3<TScalar> const, kOptions>>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar const* coeffs) : unit_quaternion_(coeffs) {}

  /// Accessor of unit quaternion.
  ///
  SOPHUS_FUNC
  [[nodiscard]] Map<Eigen::Quaternion<Scalar> const, kOptions> const&
  unitQuaternion() const {
    return unit_quaternion_;
  }

 protected:
  /// Mutator of unit_quaternion is protected to ensure class invariant.
  ///
  Map<Eigen::Quaternion<Scalar> const, kOptions> unit_quaternion_;  // NOLINT
};
}  // namespace Eigen
