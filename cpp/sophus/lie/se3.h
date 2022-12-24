// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Special Euclidean group SE(3) - rotation and translation in 3d.

#pragma once

#include "so3.h"

namespace sophus {
template <class TScalar>
class Se3;
using Se3F64 = Se3<double>;
using Se3F32 = Se3<float>;

template <class TScalar>
/* [[deprecated]] */ using SE3 = Se3<TScalar>;
/* [[deprecated]] */ using SE3d = Se3F64;
/* [[deprecated]] */ using SE3f = Se3F32;
}  // namespace sophus

namespace Eigen {  // NOLINT
namespace internal {

template <class TScalar>
struct traits<sophus::Se3<TScalar>> {
  using Scalar = TScalar;
  using TranslationType = Eigen::Matrix<Scalar, 3, 1>;
  using So3Type = sophus::So3<Scalar>;
};

template <class TScalar>
struct traits<Map<sophus::Se3<TScalar>>> : traits<sophus::Se3<TScalar>> {
  using Scalar = TScalar;
  using TranslationType = Map<Eigen::Vector3<Scalar>>;
  using So3Type = Map<sophus::So3<Scalar>>;
};

template <class TScalar>
struct traits<Map<sophus::Se3<TScalar> const>>
    : traits<sophus::Se3<TScalar> const> {
  using Scalar = TScalar;
  using TranslationType = Map<Eigen::Vector3<Scalar> const>;
  using So3Type = Map<sophus::So3<Scalar> const>;
};
}  // namespace internal
}  // namespace Eigen

namespace sophus {

/// Se3 base type - implements Se3 class but is storage agnostic.
///
/// SE(3) is the group of rotations  and translation in 3d. It is the
/// semi-direct product of SO(3) and the 3d Euclidean vector space.  The class
/// is represented using a composition of So3  for rotation and a one 3-vector
/// for translation.
///
/// SE(3) is neither compact, nor a commutative group.
///
///
///  - 4x4 Eigen::Matrix representation:
///
/// ```
///   | R t |
///   | o 1 |
/// ```
///
/// where ``R`` is a 3x3 rotation matrix, ``t`` a translation 3-vector and
/// ``o`` a 3-column vector of zeros.
///
///  - Tangent 6-vector: [upsilon, omega],
///
/// where ``upsilon`` is the translational velocity 3-vector and ``omega`` the
/// rotational velocity 3-vector.
///
///  - Internal 7-representation: [t0, t1, t2, qi0, qi1, qi2, qr],
///
/// with ``t0, t1, t2`` are the translational components, and ``qi0, qi1, q2``
/// the imaginary vector part and ``qr1`` the real part of a unit-length
/// quaternion.
///
///
/// See So3 for more details of the rotation representation in 3d.
///
template <class TDerived>
class Se3Base {
 public:
  using Scalar = typename Eigen::internal::traits<TDerived>::Scalar;
  using TranslationType =
      typename Eigen::internal::traits<TDerived>::TranslationType;
  using So3Type = typename Eigen::internal::traits<TDerived>::So3Type;
  using QuaternionType = typename So3Type::QuaternionType;
  /// Degrees of freedom of manifold, number of dimensions in tangent space
  /// (three for translation, three for rotation).
  static int constexpr kDoF = 6;
  /// Number of internal parameters used (4-tuple for quaternion, three for
  /// translation).
  static int constexpr kNumParams = 7;
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
  /// double scalars with Se3 operations.
  template <class TOtherDerived>
  using ReturnScalar = typename Eigen::
      ScalarBinaryOpTraits<Scalar, typename TOtherDerived::Scalar>::ReturnType;

  template <class TOtherDerived>
  using Se3Product = Se3<ReturnScalar<TOtherDerived>>;

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
    Eigen::Matrix3<Scalar> const mat_r = so3().matrix();
    Adjoint res;
    res.block(0, 0, 3, 3) = mat_r;
    res.block(3, 3, 3, 3) = mat_r;
    res.block(0, 3, 3, 3) = So3<Scalar>::hat(translation()) * mat_r;
    res.block(3, 0, 3, 3) = Eigen::Matrix3<Scalar>::Zero(3, 3);
    return res;
  }

  /// Extract rotation angle about canonical X-axis
  ///
  [[nodiscard]] Scalar angleX() const { return so3().angleX(); }

  /// Extract rotation angle about canonical Y-axis
  ///
  [[nodiscard]] Scalar angleY() const { return so3().angleY(); }

  /// Extract rotation angle about canonical Z-axis
  ///
  [[nodiscard]] Scalar angleZ() const { return so3().angleZ(); }

  /// Returns copy of instance casted to NewScalarType.
  ///
  template <class TNewScalarType>
  SOPHUS_FUNC [[nodiscard]] Se3<TNewScalarType> cast() const {
    return Se3<TNewScalarType>(
        so3().template cast<TNewScalarType>(),
        translation().template cast<TNewScalarType>());
  }

  /// Returns derivative of  this * exp(x)  wrt x at x=0.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kNumParams, kDoF>
  dxThisMulExpXAt0() const {
    Eigen::Matrix<Scalar, kNumParams, kDoF> j;
    Eigen::Quaternion<Scalar> const q = unitQuaternion();
    Scalar const c0 = Scalar(0.5) * q.w();
    Scalar const c1 = Scalar(0.5) * q.z();
    Scalar const c2 = -c1;
    Scalar const c3 = Scalar(0.5) * q.y();
    Scalar const c4 = Scalar(0.5) * q.x();
    Scalar const c5 = -c4;
    Scalar const c6 = -c3;
    Scalar const c7 = q.w() * q.w();
    Scalar const c8 = q.x() * q.x();
    Scalar const c9 = q.y() * q.y();
    Scalar const c10 = -c9;
    Scalar const c11 = q.z() * q.z();
    Scalar const c12 = -c11;
    Scalar const c13 = Scalar(2) * q.w();
    Scalar const c14 = c13 * q.z();
    Scalar const c15 = Scalar(2) * q.x();
    Scalar const c16 = c15 * q.y();
    Scalar const c17 = c13 * q.y();
    Scalar const c18 = c15 * q.z();
    Scalar const c19 = c7 - c8;
    Scalar const c20 = c13 * q.x();
    Scalar const c21 = Scalar(2) * q.y() * q.z();
    j(0, 0) = 0;
    j(0, 1) = 0;
    j(0, 2) = 0;
    j(0, 3) = c0;
    j(0, 4) = c2;
    j(0, 5) = c3;
    j(1, 0) = 0;
    j(1, 1) = 0;
    j(1, 2) = 0;
    j(1, 3) = c1;
    j(1, 4) = c0;
    j(1, 5) = c5;
    j(2, 0) = 0;
    j(2, 1) = 0;
    j(2, 2) = 0;
    j(2, 3) = c6;
    j(2, 4) = c4;
    j(2, 5) = c0;
    j(3, 0) = 0;
    j(3, 1) = 0;
    j(3, 2) = 0;
    j(3, 3) = c5;
    j(3, 4) = c6;
    j(3, 5) = c2;
    j(4, 0) = c10 + c12 + c7 + c8;
    j(4, 1) = -c14 + c16;
    j(4, 2) = c17 + c18;
    j(4, 3) = 0;
    j(4, 4) = 0;
    j(4, 5) = 0;
    j(5, 0) = c14 + c16;
    j(5, 1) = c12 + c19 + c9;
    j(5, 2) = -c20 + c21;
    j(5, 3) = 0;
    j(5, 4) = 0;
    j(5, 5) = 0;
    j(6, 0) = -c17 + c18;
    j(6, 1) = c20 + c21;
    j(6, 2) = c10 + c11 + c19;
    j(6, 3) = 0;
    j(6, 4) = 0;
    j(6, 5) = 0;
    return j;
  }

  /// Returns derivative of log(this^{-1} * x) by x at x=this.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, kDoF, kNumParams>
  dxLogThisInvTimesXAtThis() const {
    Eigen::Matrix<Scalar, kDoF, kNumParams> j;
    j.template block<3, 4>(0, 0).setZero();
    j.template block<3, 3>(0, 4) = so3().inverse().matrix();
    j.template block<3, 4>(3, 0) = so3().dxLogThisInvTimesXAtThis();
    j.template block<3, 3>(3, 4).setZero();
    return j;
  }

  /// Returns group inverse.
  ///
  SOPHUS_FUNC [[nodiscard]] Se3<Scalar> inverse() const {
    So3<Scalar> mat_r_inv = so3().inverse();
    return Se3<Scalar>(mat_r_inv, mat_r_inv * (translation() * Scalar(-1)));
  }

  /// Logarithmic map
  ///
  /// Computes the logarithm, the inverse of the group exponential which maps
  /// element of the group (rigid body transformations) to elements of the
  /// tangent space (twist).
  ///
  /// To be specific, this function computes ``vee(logmat(.))`` with
  /// ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  /// of SE(3).
  ///
  SOPHUS_FUNC [[nodiscard]] Tangent log() const {
    // For the derivation of the logarithm of SE(3), see
    // J. Gallier, D. Xu, "Computing exponentials of skew symmetric matrices
    // and logarithms of orthogonal matrices", IJRA 2002.
    // https:///pdfs.semanticscholar.org/cfe3/e4b39de63c8cabd89bf3feff7f5449fc981d.pdf
    // (Sec. 6., pp. 8)
    using std::abs;
    using std::cos;
    using std::sin;
    Tangent vec_upsilon_omega;
    auto omega_and_theta = so3().logAndTheta();
    Scalar theta = omega_and_theta.theta;
    Eigen::Vector3<Scalar> const& vec_omega = omega_and_theta.tangent;
    vec_upsilon_omega.template tail<3>() = vec_omega;
    Eigen::Matrix3<Scalar> mat_v_inv =
        So3<Scalar>::leftJacobianInverse(vec_omega, theta);
    vec_upsilon_omega.template head<3>() = mat_v_inv * translation();
    return vec_upsilon_omega;
  }

  /// It re-normalizes the So3 element.
  ///
  /// Note: Because of the class invariant of So3, there is typically no need to
  /// call this function directly.
  ///
  SOPHUS_FUNC void normalize() { so3().normalize(); }

  /// Returns 4x4 matrix representation of the instance.
  ///
  /// It has the following form:
  ///
  /// ```
  ///   | R t |
  ///   | o 1 |
  /// ```
  ///
  /// where ``R`` is a 3x3 rotation matrix, ``t`` a translation 3-vector and
  /// ``o`` a 3-column vector of zeros.
  ///
  SOPHUS_FUNC [[nodiscard]] Transformation matrix() const {
    Transformation homogeneous_matrix;
    homogeneous_matrix.template topLeftCorner<3, 4>() = matrix3x4();
    homogeneous_matrix.row(3) =
        Eigen::Matrix<Scalar, 1, 4>(Scalar(0), Scalar(0), Scalar(0), Scalar(1));
    return homogeneous_matrix;
  }

  /// Returns the significant first three rows of the matrix above.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix<Scalar, 3, 4> matrix3x4() const {
    Eigen::Matrix<Scalar, 3, 4> matrix;
    matrix.template topLeftCorner<3, 3>() = rotationMatrix();
    matrix.col(3) = translation();
    return matrix;
  }

  /// Assignment-like operator from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Se3Base<TDerived>& operator=(
      Se3Base<TOtherDerived> const& other) {
    so3() = other.so3();
    translation() = other.translation();
    return *this;
  }

  /// Group multiplication, which is rotation concatenation.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Se3Product<TOtherDerived> operator*(
      Se3Base<TOtherDerived> const& other) const {
    return Se3Product<TOtherDerived>(
        so3() * other.so3(), translation() + so3() * other.translation());
  }

  /// Group action on 3-points.
  ///
  /// This function rotates and translates a three dimensional point ``p`` by
  /// the SE(3) element ``bar_transform_foo = (bar_R_foo, t_bar)`` (= rigid body
  /// transformation):
  ///
  ///   ``p_bar = bar_R_foo * p_foo + t_bar``.
  ///
  template <
      typename TPointDerived,
      typename = typename std::enable_if<
          IsFixedSizeVector<TPointDerived, 3>::value>::type>
  SOPHUS_FUNC PointProduct<TPointDerived> operator*(
      Eigen::MatrixBase<TPointDerived> const& p) const {
    return so3() * p + translation();
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
        so3() * p.template head<3>() + p(3) * translation();
    return HomogeneousPointProduct<THPointDerived>(tp(0), tp(1), tp(2), p(3));
  }

  /// Group action on lines.
  ///
  /// This function rotates and translates a parametrized line
  /// ``l(t) = o + t * d`` by the SE(3) element:
  ///
  /// Origin is transformed using SE(3) action
  /// Direction is transformed using rotation part
  ///
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), so3() * l.direction());
  }

  /// Group action on planes.
  ///
  /// This function rotates and translates a plane
  /// ``n.x + d = 0`` by the SE(3) element:
  ///
  /// Normal vector ``n`` is rotated
  /// Offset ``d`` is adjusted for translation
  ///
  SOPHUS_FUNC Hyperplane operator*(Hyperplane const& p) const {
    Hyperplane const rotated = so3() * p;
    return Hyperplane(
        rotated.normal(),
        rotated.offset() - translation().dot(rotated.normal()));
  }

  /// In-place group multiplication. This method is only valid if the return
  /// type of the multiplication is compatible with this Se3's Scalar type.
  ///
  template <
      typename TOtherDerived,
      typename = typename std::enable_if<
          std::is_same<Scalar, ReturnScalar<TOtherDerived>>::value>::type>
  SOPHUS_FUNC Se3Base<TDerived>& operator*=(
      Se3Base<TOtherDerived> const& other) {
    *static_cast<TDerived*>(this) = *this * other;
    return *this;
  }

  /// Returns rotation matrix.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Matrix3<Scalar> rotationMatrix() const {
    return so3().matrix();
  }

  /// Mutator of So3 group.
  ///
  SOPHUS_FUNC So3Type& so3() { return static_cast<TDerived*>(this)->so3(); }

  /// Accessor of So3 group.
  ///
  SOPHUS_FUNC [[nodiscard]] So3Type const& so3() const {
    return static_cast<TDerived const*>(this)->so3();
  }

  /// Takes in quaternion, and normalizes it.
  ///
  /// Precondition: The quaternion must not be close to zero.
  ///
  SOPHUS_FUNC void setQuaternion(Eigen::Quaternion<Scalar> const& quat) {
    so3().setQuaternion(quat);
  }

  /// Sets ``so3`` using ``rotation_matrix``.
  ///
  /// Precondition: ``R`` must be orthogonal and ``det(R)=1``.
  ///-
  SOPHUS_FUNC void setRotationMatrix(Eigen::Matrix3<Scalar> const& mat_r) {
    FARM_CHECK(isOrthogonal(mat_r), "R is not orthogonal:\n {}", mat_r);
    FARM_CHECK(
        mat_r.determinant() > Scalar(0),
        "det(R) is not positive: {}",
        mat_r.determinant());
    so3().setQuaternion(Eigen::Quaternion<Scalar>(mat_r));
  }

  /// Returns internal parameters of SE(3).
  ///
  /// It returns (q.imag[0], q.imag[1], q.imag[2], q.real, t[0], t[1], t[2]),
  /// with q being the unit quaternion, t the translation 3-vector.
  ///
  SOPHUS_FUNC [[nodiscard]] Eigen::Vector<Scalar, kNumParams> params() const {
    Eigen::Vector<Scalar, kNumParams> p;
    p << so3().params(), translation();
    return p;
  }

  SOPHUS_FUNC void setParams(Eigen::Vector<Scalar, kNumParams> const& params) {
    this->translation() = params.template tail<3>();
    this->so3().setParams(params.template head<4>());
  }

  /// Mutator of translation vector.
  ///
  SOPHUS_FUNC TranslationType& translation() {
    return static_cast<TDerived*>(this)->translation();
  }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] TranslationType const& translation() const {
    return static_cast<TDerived const*>(this)->translation();
  }

  /// Accessor of unit quaternion.
  ///
  SOPHUS_FUNC [[nodiscard]] QuaternionType const& unitQuaternion() const {
    return this->so3().unitQuaternion();
  }
};

/// Se3 using default storage; derived from Se3Base.
template <class TScalar>
class Se3 : public Se3Base<Se3<TScalar>> {
  using Base = Se3Base<Se3<TScalar>>;

 public:
  static int constexpr kDoF = Base::kDoF;
  static int constexpr kNumParams = Base::kNumParams;

  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using SO3Member = So3<Scalar>;
  using TranslationMember = Eigen::Matrix<Scalar, 3, 1>;

  using Base::operator=;

  /// Define copy-assignment operator explicitly. The definition of
  /// implicit copy assignment operator is deprecated in presence of a
  /// user-declared copy constructor (-Wdeprecated-copy in clang >= 13).
  SOPHUS_FUNC Se3& operator=(Se3 const& other) = default;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Default constructor initializes rigid body motion to the identity.
  ///
  SOPHUS_FUNC Se3();

  /// Copy constructor
  ///
  SOPHUS_FUNC Se3(Se3 const& other) = default;

  /// Copy-like constructor from OtherDerived.
  ///
  template <class TOtherDerived>
  SOPHUS_FUNC Se3(Se3Base<TOtherDerived> const& other)
      : so3_(other.so3()), translation_(other.translation()) {
    static_assert(
        std::is_same<typename TOtherDerived::Scalar, Scalar>::value,
        "must be same Scalar type");
  }

  /// Constructor from So3 and translation vector
  ///
  template <class TOtherDerived, class TD>
  SOPHUS_FUNC Se3(
      So3Base<TOtherDerived> const& so3,
      Eigen::MatrixBase<TD> const& translation)
      : so3_(so3), translation_(translation) {
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
  ///               of 1.
  ///
  SOPHUS_FUNC
  Se3(Eigen::Matrix3<Scalar> const& rotation_matrix, Point const& translation)
      : so3_(rotation_matrix), translation_(translation) {}

  /// Constructor from quaternion and translation vector.
  ///
  /// Precondition: ``quaternion`` must not be close to zero.
  ///
  SOPHUS_FUNC Se3(
      Eigen::Quaternion<Scalar> const& quaternion, Point const& translation)
      : so3_(quaternion), translation_(translation) {}

  /// Constructor from 4x4 matrix
  ///
  /// Precondition: Rotation matrix needs to be orthogonal with determinant
  ///               of 1. The last row must be ``(0, 0, 0, 1)``.
  ///
  SOPHUS_FUNC explicit Se3(Eigen::Matrix4<Scalar> const& mat_t)
      : so3_(mat_t.template topLeftCorner<3, 3>()),
        translation_(mat_t.template block<3, 1>(0, 3)) {
    FARM_CHECK(
        (mat_t.row(3) - Eigen::Matrix<Scalar, 1, 4>(
                            Scalar(0), Scalar(0), Scalar(0), Scalar(1)))
                .squaredNorm() < kEpsilon<Scalar>,
        "Last row is not (0,0,0,1), but ({}).",
        mat_t.row(3));
  }

  /// This provides unsafe read/write access to internal data. SO(3) is
  /// represented by an Eigen::Quaternion (four parameters). When using direct
  /// write access, the user needs to take care of that the quaternion stays
  /// normalized.
  ///
  SOPHUS_FUNC Scalar* data() {
    // so3_ and translation_ are laid out sequentially with no padding
    return so3_.data();
  }

  /// Const version of data() above.
  ///
  SOPHUS_FUNC [[nodiscard]] Scalar const* data() const {
    // so3_ and translation_ are laid out sequentially with no padding
    return so3_.data();
  }

  /// Mutator of So3
  ///
  SOPHUS_FUNC SO3Member& so3() { return so3_; }

  /// Accessor of So3
  ///
  SOPHUS_FUNC [[nodiscard]] SO3Member const& so3() const { return so3_; }

  /// Mutator of translation vector
  ///
  SOPHUS_FUNC TranslationMember& translation() { return translation_; }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] TranslationMember const& translation() const {
    return translation_;
  }

  SOPHUS_FUNC static Eigen::Matrix3<Scalar> jacobianUpperRightBlock(
      Eigen::Vector3<Scalar> const& vec_upsilon,
      Eigen::Vector3<Scalar> const& vec_omega) {
    using std::cos;
    using std::sin;
    using std::sqrt;

    Scalar const half(0.5);

    Scalar const theta_sq = vec_omega.squaredNorm();
    Eigen::Matrix3<Scalar> const mat_upsilon = So3<Scalar>::hat(vec_upsilon);

    Eigen::Matrix3<Scalar> q;
    if (theta_sq < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      q = half * mat_upsilon;

    } else {
      Scalar const theta = sqrt(theta_sq);
      Scalar const i_theta = Scalar(1) / theta;
      Scalar const i_theta_sq = i_theta * i_theta;
      Scalar const i_theta_po4 = i_theta_sq * i_theta_sq;
      Scalar const st = sin(theta);
      Scalar const ct = cos(theta);
      Scalar const c1 = i_theta_sq - st * i_theta_sq * i_theta;
      Scalar const c2 = half * i_theta_sq + ct * i_theta_po4 - i_theta_po4;
      Scalar const c3 = i_theta_po4 + half * ct * i_theta_po4 -
                        Scalar(1.5) * st * i_theta * i_theta_po4;

      Eigen::Matrix3<Scalar> const mat_omega = So3<Scalar>::hat(vec_omega);
      Eigen::Matrix3<Scalar> const mat_omega_upsilon = mat_omega * mat_upsilon;
      Eigen::Matrix3<Scalar> const mat_omega_upsilon_omega =
          mat_omega_upsilon * mat_omega;
      q = half * mat_upsilon +
          c1 * (mat_omega_upsilon + mat_upsilon * mat_omega +
                mat_omega_upsilon_omega) -
          c2 * (theta_sq * mat_upsilon + Scalar(2) * mat_omega_upsilon_omega) +
          c3 * (mat_omega_upsilon_omega * mat_omega +
                mat_omega * mat_omega_upsilon_omega);
    }
    return q;
  }

  SOPHUS_FUNC static Eigen::Matrix<Scalar, kDoF, kDoF> leftJacobian(
      Tangent const& vec_upsilon_omega) {
    Eigen::Vector3<Scalar> const vec_upsilon =
        vec_upsilon_omega.template head<3>();
    Eigen::Vector3<Scalar> const vec_omega =
        vec_upsilon_omega.template tail<3>();
    Eigen::Matrix3<Scalar> const mat_j = So3<Scalar>::leftJacobian(vec_omega);
    Eigen::Matrix3<Scalar> mat_q =
        jacobianUpperRightBlock(vec_upsilon, vec_omega);
    Eigen::Matrix<Scalar, 6, 6> mat_u;
    mat_u << mat_j, mat_q, Eigen::Matrix3<Scalar>::Zero(), mat_j;
    return mat_u;
  }

  SOPHUS_FUNC static Eigen::Matrix<Scalar, kDoF, kDoF> leftJacobianInverse(
      Tangent const& vec_upsilon_omega) {
    Eigen::Vector3<Scalar> const vec_upsilon =
        vec_upsilon_omega.template head<3>();
    Eigen::Vector3<Scalar> const vec_omega =
        vec_upsilon_omega.template tail<3>();
    Eigen::Matrix3<Scalar> const mat_j_inv =
        So3<Scalar>::leftJacobianInverse(vec_omega);
    Eigen::Matrix3<Scalar> mat_q =
        jacobianUpperRightBlock(vec_upsilon, vec_omega);
    Eigen::Matrix<Scalar, 6, 6> mat_u;
    mat_u << mat_j_inv, -mat_j_inv * mat_q * mat_j_inv,
        Eigen::Matrix3<Scalar>::Zero(), mat_j_inv;
    return mat_u;
  }

  /// Returns derivative of exp(x) wrt. x.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParams, kDoF> dxExpX(
      Tangent const& upsilon_omega) {
    using std::cos;
    using std::pow;
    using std::sin;
    using std::sqrt;
    Eigen::Matrix<Scalar, kNumParams, kDoF> j;
    Eigen::Vector<Scalar, 3> upsilon = upsilon_omega.template head<3>();
    Eigen::Vector<Scalar, 3> omega = upsilon_omega.template tail<3>();

    Scalar const c0 = omega[0] * omega[0];
    Scalar const c1 = omega[1] * omega[1];
    Scalar const c2 = omega[2] * omega[2];
    Scalar const c3 = c0 + c1 + c2;
    Scalar const o(0);
    Scalar const h(0.5);
    Scalar const i(1);

    if (c3 < kEpsilon<Scalar>) {
      Scalar const ux = Scalar(0.5) * upsilon[0];
      Scalar const uy = Scalar(0.5) * upsilon[1];
      Scalar const uz = Scalar(0.5) * upsilon[2];

      /// clang-format off
      j << o, o, o, h, o, o, o, o, o, o, h, o, o, o, o, o, o, h, o, o, o, o, o,
          o, i, o, o, o, uz, -uy, o, i, o, -uz, o, ux, o, o, i, uy, -ux, o;
      /// clang-format on
      return j;
    }

    Scalar const c4 = sqrt(c3);
    Scalar const c5 = Scalar(1.0) / c4;
    Scalar const c6 = Scalar(0.5) * c4;
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
    Scalar const c20 = c5 * omega[0];
    Scalar const c21 = Scalar(0.5) * c7;
    Scalar const c22 = c5 * omega[1];
    Scalar const c23 = c5 * omega[2];
    Scalar const c24 = -c1;
    Scalar const c25 = -c2;
    Scalar const c26 = c24 + c25;
    Scalar const c27 = sin(c4);
    Scalar const c28 = -c27 + c4;
    Scalar const c29 = c28 * c9;
    Scalar const c30 = cos(c4);
    Scalar const c31 = -c30 + Scalar(1);
    Scalar const c32 = c11 * c31;
    Scalar const c33 = c32 * omega[2];
    Scalar const c34 = c29 * omega[0];
    Scalar const c35 = c34 * omega[1];
    Scalar const c36 = c32 * omega[1];
    Scalar const c37 = c34 * omega[2];
    Scalar const c38 = pow(c3, -5.0L / 2.0L);
    Scalar const c39 = Scalar(3) * c28 * c38 * omega[0];
    Scalar const c40 = c26 * c9;
    Scalar const c41 = -c20 * c30 + c20;
    Scalar const c42 = c27 * c9 * omega[0];
    Scalar const c43 = c42 * omega[1];
    Scalar const c44 = pow(c3, -2);
    Scalar const c45 = Scalar(2) * c31 * c44 * omega[0];
    Scalar const c46 = c45 * omega[1];
    Scalar const c47 = c29 * omega[2];
    Scalar const c48 = c43 - c46 + c47;
    Scalar const c49 = Scalar(3) * c0 * c28 * c38;
    Scalar const c50 = c9 * omega[0] * omega[2];
    Scalar const c51 = c41 * c50 - c49 * omega[2];
    Scalar const c52 = c9 * omega[0] * omega[1];
    Scalar const c53 = c41 * c52 - c49 * omega[1];
    Scalar const c54 = c42 * omega[2];
    Scalar const c55 = c45 * omega[2];
    Scalar const c56 = c29 * omega[1];
    Scalar const c57 = -c54 + c55 + c56;
    Scalar const c58 = Scalar(-2) * c56;
    Scalar const c59 = Scalar(3) * c28 * c38 * omega[1];
    Scalar const c60 = -c22 * c30 + c22;
    Scalar const c61 = -c18 * c39;
    Scalar const c62 = c32 + c61;
    Scalar const c63 = c27 * c9;
    Scalar const c64 = c1 * c63;
    Scalar const c65 = Scalar(2) * c31 * c44;
    Scalar const c66 = c1 * c65;
    Scalar const c67 = c50 * c60;
    Scalar const c68 = -c1 * c39 + c52 * c60;
    Scalar const c69 = c18 * c63;
    Scalar const c70 = c18 * c65;
    Scalar const c71 = c34 - c69 + c70;
    Scalar const c72 = Scalar(-2) * c47;
    Scalar const c73 = Scalar(3) * c28 * c38 * omega[2];
    Scalar const c74 = -c23 * c30 + c23;
    Scalar const c75 = -c32 + c61;
    Scalar const c76 = c2 * c63;
    Scalar const c77 = c2 * c65;
    Scalar const c78 = c52 * c74;
    Scalar const c79 = c34 + c69 - c70;
    Scalar const c80 = -c2 * c39 + c50 * c74;
    Scalar const c81 = -c0;
    Scalar const c82 = c25 + c81;
    Scalar const c83 = c32 * omega[0];
    Scalar const c84 = c18 * c29;
    Scalar const c85 = Scalar(-2) * c34;
    Scalar const c86 = c82 * c9;
    Scalar const c87 = c0 * c63;
    Scalar const c88 = c0 * c65;
    Scalar const c89 = c9 * omega[1] * omega[2];
    Scalar const c90 = c41 * c89;
    Scalar const c91 = c54 - c55 + c56;
    Scalar const c92 = -c1 * c73 + c60 * c89;
    Scalar const c93 = -c43 + c46 + c47;
    Scalar const c94 = -c2 * c59 + c74 * c89;
    Scalar const c95 = c24 + c81;
    Scalar const c96 = c9 * c95;
    j(0, 0) = o;
    j(0, 1) = o;
    j(0, 2) = o;
    j(0, 3) = -c0 * c10 + c0 * c13 + c8;
    j(0, 4) = c16;
    j(0, 5) = c17;
    j(1, 0) = o;
    j(1, 1) = o;
    j(1, 2) = o;
    j(1, 3) = c16;
    j(1, 4) = -c1 * c10 + c1 * c13 + c8;
    j(1, 5) = c19;
    j(2, 0) = o;
    j(2, 1) = o;
    j(2, 2) = o;
    j(2, 3) = c17;
    j(2, 4) = c19;
    j(2, 5) = -c10 * c2 + c13 * c2 + c8;
    j(3, 0) = o;
    j(3, 1) = o;
    j(3, 2) = o;
    j(3, 3) = -c20 * c21;
    j(3, 4) = -c21 * c22;
    j(3, 5) = -c21 * c23;
    j(4, 0) = c26 * c29 + Scalar(1);
    j(4, 1) = -c33 + c35;
    j(4, 2) = c36 + c37;
    j(4, 3) = upsilon[0] * (-c26 * c39 + c40 * c41) + upsilon[1] * (c53 + c57) +
              upsilon[2] * (c48 + c51);
    j(4, 4) = upsilon[0] * (-c26 * c59 + c40 * c60 + c58) +
              upsilon[1] * (c68 + c71) + upsilon[2] * (c62 + c64 - c66 + c67);
    j(4, 5) = upsilon[0] * (-c26 * c73 + c40 * c74 + c72) +
              upsilon[1] * (c75 - c76 + c77 + c78) + upsilon[2] * (c79 + c80);
    j(5, 0) = c33 + c35;
    j(5, 1) = c29 * c82 + Scalar(1);
    j(5, 2) = -c83 + c84;
    j(5, 3) = upsilon[0] * (c53 + c91) +
              upsilon[1] * (-c39 * c82 + c41 * c86 + c85) +
              upsilon[2] * (c75 - c87 + c88 + c90);
    j(5, 4) = upsilon[0] * (c68 + c79) + upsilon[1] * (-c59 * c82 + c60 * c86) +
              upsilon[2] * (c92 + c93);
    j(5, 5) = upsilon[0] * (c62 + c76 - c77 + c78) +
              upsilon[1] * (c72 - c73 * c82 + c74 * c86) +
              upsilon[2] * (c57 + c94);
    j(6, 0) = -c36 + c37;
    j(6, 1) = c83 + c84;
    j(6, 2) = c29 * c95 + Scalar(1);
    j(6, 3) = upsilon[0] * (c51 + c93) + upsilon[1] * (c62 + c87 - c88 + c90) +
              upsilon[2] * (-c39 * c95 + c41 * c96 + c85);
    j(6, 4) = upsilon[0] * (-c64 + c66 + c67 + c75) + upsilon[1] * (c48 + c92) +
              upsilon[2] * (c58 - c59 * c95 + c60 * c96);
    j(6, 5) = upsilon[0] * (c71 + c80) + upsilon[1] * (c91 + c94) +
              upsilon[2] * (-c73 * c95 + c74 * c96);

    return j;
  }

  /// Returns derivative of exp(x) wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, kNumParams, kDoF> dxExpXAt0() {
    Eigen::Matrix<Scalar, kNumParams, kDoF> dx;
    Scalar const o(0);
    Scalar const h(0.5);
    Scalar const i(1);

    // clang-format off
    dx << o, o, o, h, o, o,
          o, o, o, o, h, o,
          o, o, o, o, o, h,
	        o, o, o, o, o, o,
	        i, o, o, o, o, o,
	        o, i, o, o, o, o,
	        o, o, i, o, o, o;
    // clang-format on
    return dx;
  }

  /// Returns derivative of ``expmat(x)`` wrt. ``x_i at x=0``, with
  /// ``expmat(.)`` being the matrix exponential.
  ///
  SOPHUS_FUNC static Transformation dxiExpmatXAt0(int i) {
    return generator(i);
  }

  /// Returns derivative of exp(x) * p wrt. x_i at x=0.
  ///
  SOPHUS_FUNC static Eigen::Matrix<Scalar, 3, kDoF> dxExpXTimesPointAt0(
      Point const& point) {
    Eigen::Matrix<Scalar, 3, kDoF> j;
    j << Eigen::Matrix3<Scalar>::Identity(),
        sophus::So3<Scalar>::dxExpXTimesPointAt0(point);
    return j;
  }

  /// Group exponential
  ///
  /// This functions takes in an element of tangent space (= twist ``a``) and
  /// returns the corresponding element of the group SE(3).
  ///
  /// The first three components of ``a`` represent the translational part
  /// ``upsilon`` in the tangent space of SE(3), while the last three components
  /// of ``a`` represents the rotation vector ``omega``.
  /// To be more specific, this function computes ``expmat(hat(a))`` with
  /// ``expmat(.)`` being the matrix exponential and ``hat(.)`` the hat-operator
  /// of SE(3), see below.
  ///
  SOPHUS_FUNC static Se3<Scalar> exp(Tangent const& vec_a) {
    using std::cos;
    using std::sin;
    Eigen::Vector3<Scalar> const vec_omega = vec_a.template tail<3>();

    typename So3<Scalar>::So3AndTheta const so3_and_theta =
        So3<Scalar>::expAndTheta(vec_omega);
    Eigen::Matrix3<Scalar> const mat_a =
        So3<Scalar>::leftJacobian(vec_omega, so3_and_theta.theta);
    return Se3<Scalar>(so3_and_theta.so3, mat_a * vec_a.template head<3>());
  }

  /// Returns closest Se3 given arbitrary 4x4 matrix.
  ///
  template <class TS = Scalar>
  SOPHUS_FUNC static std::enable_if_t<std::is_floating_point<TS>::value, Se3>
  fitToSe3(Eigen::Matrix4<Scalar> const& t) {
    return Se3(
        So3<Scalar>::fitToSo3(t.template block<3, 3>(0, 0)),
        t.template block<3, 1>(0, 3));
  }

  /// Returns the ith infinitesimal generators of SE(3).
  ///
  /// The infinitesimal generators of SE(3) are:
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
  /// ```
  ///
  /// Precondition: ``i`` must be in [0, 5].
  ///
  SOPHUS_FUNC static Transformation generator(int i) {
    FARM_CHECK(i >= 0 && i <= 5, "i should be in range [0,5].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  /// hat-operator
  ///
  /// It takes in the 6-vector representation (= twist) and returns the
  /// corresponding matrix representation of Lie algebra element.
  ///
  /// Formally, the hat()-operator of SE(3) is defined as
  ///
  ///   ``hat(.): R^6 -> R^{4x4},  hat(a) = sum_i a_i * G_i``  (for i=0,...,5)
  ///
  /// with ``G_i`` being the ith infinitesimal generator of SE(3).
  ///
  /// The corresponding inverse is the vee()-operator, see below.
  ///
  SOPHUS_FUNC static Transformation hat(Tangent const& a) {
    Transformation omega;
    omega.setZero();
    omega.template topLeftCorner<3, 3>() =
        So3<Scalar>::hat(a.template tail<3>());
    omega.col(3).template head<3>() = a.template head<3>();
    return omega;
  }

  /// Lie bracket
  ///
  /// It computes the Lie bracket of SE(3). To be more specific, it computes
  ///
  ///   ``[omega_1, omega_2]_se3 := vee([hat(omega_1), hat(omega_2)])``
  ///
  /// with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.)`` the
  /// hat()-operator and ``vee(.)`` the vee()-operator of SE(3).
  ///
  SOPHUS_FUNC static Tangent lieBracket(Tangent const& a, Tangent const& b) {
    Eigen::Vector3<Scalar> const upsilon1 = a.template head<3>();
    Eigen::Vector3<Scalar> const upsilon2 = b.template head<3>();
    Eigen::Vector3<Scalar> const omega1 = a.template tail<3>();
    Eigen::Vector3<Scalar> const omega2 = b.template tail<3>();

    Tangent res;
    res.template head<3>() = omega1.cross(upsilon2) + upsilon1.cross(omega2);
    res.template tail<3>() = omega1.cross(omega2);

    return res;
  }

  /// Construct x-axis rotation.
  ///
  static SOPHUS_FUNC Se3 rotX(Scalar const& x) {
    return Se3(So3<Scalar>::rotX(x), Eigen::Vector3<Scalar>::Zero());
  }

  /// Construct y-axis rotation.
  ///
  static SOPHUS_FUNC Se3 rotY(Scalar const& y) {
    return Se3(So3<Scalar>::rotY(y), Eigen::Vector3<Scalar>::Zero());
  }

  /// Construct z-axis rotation.
  ///
  static SOPHUS_FUNC Se3 rotZ(Scalar const& z) {
    return Se3(So3<Scalar>::rotZ(z), Eigen::Vector3<Scalar>::Zero());
  }

  /// Draw uniform sample from SE(3) manifold.
  ///
  /// Translations are drawn component-wise from the range [-1, 1].
  ///
  template <class TUniformRandomBitGenerator>
  static Se3 sampleUniform(TUniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    return Se3(
        So3<Scalar>::sampleUniform(generator),
        Eigen::Vector3<Scalar>(
            uniform(generator), uniform(generator), uniform(generator)));
  }

  /// Construct a translation only Se3 instance.
  ///
  template <class TX, class TY, class TZ>
  static SOPHUS_FUNC Se3 trans(TX const& x, TY const& y, TZ const& z) {
    return Se3(So3<Scalar>(), Eigen::Vector3<Scalar>(x, y, z));
  }

  static SOPHUS_FUNC Se3 trans(Eigen::Vector3<Scalar> const& xyz) {
    return Se3(So3<Scalar>(), xyz);
  }

  /// Construct x-axis translation.
  ///
  static SOPHUS_FUNC Se3 transX(Scalar const& x) {
    return Se3::trans(x, Scalar(0), Scalar(0));
  }

  /// Construct y-axis translation.
  ///
  static SOPHUS_FUNC Se3 transY(Scalar const& y) {
    return Se3::trans(Scalar(0), y, Scalar(0));
  }

  /// Construct z-axis translation.
  ///
  static SOPHUS_FUNC Se3 transZ(Scalar const& z) {
    return Se3::trans(Scalar(0), Scalar(0), z);
  }

  /// vee-operator
  ///
  /// It takes 4x4-matrix representation ``Omega`` and maps it to the
  /// corresponding 6-vector representation of Lie algebra.
  ///
  /// This is the inverse of the hat()-operator, see above.
  ///
  /// Precondition: ``Omega`` must have the following structure:
  ///
  ///                |  0 -f  e  a |
  ///                |  f  0 -d  b |
  ///                | -e  d  0  c
  ///                |  0  0  0  0 | .
  ///
  SOPHUS_FUNC static Tangent vee(Transformation const& omega) {
    Tangent upsilon_omega;
    upsilon_omega.template head<3>() = omega.col(3).template head<3>();
    upsilon_omega.template tail<3>() =
        So3<Scalar>::vee(omega.template topLeftCorner<3, 3>());
    return upsilon_omega;
  }

 protected:
  SO3Member so3_;                  // NOLINT
  TranslationMember translation_;  // NOLINT
};

template <class TScalar>
SOPHUS_FUNC Se3<TScalar>::Se3() : translation_(TranslationMember::Zero()) {
  static_assert(
      std::is_standard_layout<Se3>::value,
      "Assume standard layout for the use of offsetof check below.");
  static_assert(
      offsetof(Se3, so3_) + sizeof(Scalar) * So3<Scalar>::kNumParams ==
          offsetof(Se3, translation_),
      "This class assumes packed storage and hence will only work "
      "correctly depending on the compiler (options) - in "
      "particular when using [this->data(), this-data() + "
      "kNumParams] to access the raw data in a contiguous fashion.");
}
}  // namespace sophus

namespace Eigen {  // NOLINT

/// Specialization of Eigen::Map for ``Se3``; derived from Se3Base.
///
/// Allows us to wrap Se3 objects around POD array.
template <class TScalar>
class Map<sophus::Se3<TScalar>>
    : public sophus::Se3Base<Map<sophus::Se3<TScalar>>> {
 public:
  using Base = sophus::Se3Base<Map<sophus::Se3<TScalar>>>;
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
      : so3_(coeffs), translation_(coeffs + sophus::So3<Scalar>::kNumParams) {}

  /// Mutator of So3
  ///
  SOPHUS_FUNC Map<sophus::So3<Scalar>>& so3() { return so3_; }

  /// Accessor of So3
  ///
  SOPHUS_FUNC [[nodiscard]] Map<sophus::So3<Scalar>> const& so3() const {
    return so3_;
  }

  /// Mutator of translation vector
  ///
  SOPHUS_FUNC Map<Eigen::Matrix<Scalar, 3, 1>>& translation() {
    return translation_;
  }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Matrix<Scalar, 3, 1>> const&
  translation() const {
    return translation_;
  }

 protected:
  Map<sophus::So3<Scalar>> so3_;             // NOLINT
  Map<Eigen::Vector3<Scalar>> translation_;  // NOLINT
};

/// Specialization of Eigen::Map for ``Se3 const``; derived from Se3Base.
///
/// Allows us to wrap Se3 objects around POD array.
template <class TScalar>
class Map<sophus::Se3<TScalar> const>
    : public sophus::Se3Base<Map<sophus::Se3<TScalar> const>> {
 public:
  using Base = sophus::Se3Base<Map<sophus::Se3<TScalar> const>>;
  using Scalar = TScalar;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC explicit Map(Scalar const* coeffs)
      : so3_(coeffs), translation_(coeffs + sophus::So3<Scalar>::kNumParams) {}

  /// Accessor of So3
  ///
  SOPHUS_FUNC [[nodiscard]] Map<sophus::So3<Scalar> const> const& so3() const {
    return so3_;
  }

  /// Accessor of translation vector
  ///
  SOPHUS_FUNC [[nodiscard]] Map<Eigen::Vector3<Scalar> const> const&
  translation() const {
    return translation_;
  }

 protected:
  Map<sophus::So3<Scalar> const> so3_;             // NOLINT
  Map<Eigen::Vector3<Scalar> const> translation_;  // NOLINT
};
}  // namespace Eigen
