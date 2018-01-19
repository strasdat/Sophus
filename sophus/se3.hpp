#ifndef SOPHUS_SE3_HPP
#define SOPHUS_SE3_HPP

#include "so3.hpp"

namespace Sophus {
template <class Scalar_, int Options = 0>
class SE3;
using SE3d = SE3<double>;
using SE3f = SE3<float>;
}  // namespace Sophus

namespace Eigen {
namespace internal {

template <class Scalar_, int Options>
struct traits<Sophus::SE3<Scalar_, Options>> {
  using Scalar = Scalar_;
  using TranslationType = Sophus::Vector3<Scalar, Options>;
  using SO3Type = Sophus::SO3<Scalar, Options>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::SE3<Scalar_>, Options>>
    : traits<Sophus::SE3<Scalar_, Options>> {
  using Scalar = Scalar_;
  using TranslationType = Map<Sophus::Vector3<Scalar>, Options>;
  using SO3Type = Map<Sophus::SO3<Scalar>, Options>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::SE3<Scalar_> const, Options>>
    : traits<Sophus::SE3<Scalar_, Options> const> {
  using Scalar = Scalar_;
  using TranslationType = Map<Sophus::Vector3<Scalar> const, Options>;
  using SO3Type = Map<Sophus::SO3<Scalar> const, Options>;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {

// SE3 base type - implements SE3 class but is storage agnostic.
//
// SE(3) is the group of rotations  and translation in 3d. It is the semi-direct
// product of SO(3) and the 3d Euclidean vector space.  The class is represented
// using a composition of SO3  for rotation and a one 3-vector for
// translation.
//
// SE(3) is neither compact, nor a commutative group.
//
// See SO3 for more details of the rotation representation in 3d.
//
template <class Derived>
class SE3Base {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using TranslationType =
      typename Eigen::internal::traits<Derived>::TranslationType;
  using SO3Type = typename Eigen::internal::traits<Derived>::SO3Type;
  using QuaternionType = typename SO3Type::QuaternionType;
  // Degrees of freedom of manifold, number of dimensions in tangent space
  // (three for translation, three for rotation).
  static int constexpr DoF = 6;
  // Number of internal parameters used (4-tuple for quaternion, three for
  // translation).
  static int constexpr num_parameters = 7;
  // Group transformations are 4x4 matrices.
  static int constexpr N = 4;
  using Transformation = Matrix<Scalar, N, N>;
  using Point = Vector3<Scalar>;
  using Line = ParametrizedLine3<Scalar>;
  using Tangent = Vector<Scalar, DoF>;
  using Adjoint = Matrix<Scalar, DoF, DoF>;
  // Adjoint transformation
  //
  // This function return the adjoint transformation ``Ad`` of the group
  // element ``A`` such that for all ``x`` it holds that
  // ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  //
  SOPHUS_FUNC Adjoint Adj() const {
    Sophus::Matrix3<Scalar> const R = so3().matrix();
    Adjoint res;
    res.block(0, 0, 3, 3) = R;
    res.block(3, 3, 3, 3) = R;
    res.block(0, 3, 3, 3) = SO3<Scalar>::hat(translation()) * R;
    res.block(3, 0, 3, 3) = Matrix3<Scalar>::Zero(3, 3);
    return res;
  }

  // Extract rotation angle about canonical X-axis
  //
  Scalar angleX() const { return so3().angleX(); }

  // Extract rotation angle about canonical Y-axis
  //
  Scalar angleY() const { return so3().angleY(); }

  // Extract rotation angle about canonical Z-axis
  //
  Scalar angleZ() const { return so3().angleZ(); }

  // Returns copy of instance casted to NewScalarType.
  //
  template <class NewScalarType>
  SOPHUS_FUNC SE3<NewScalarType> cast() const {
    return SE3<NewScalarType>(so3().template cast<NewScalarType>(),
                              translation().template cast<NewScalarType>());
  }

  // Returns derivative of  this * exp(x)  wrt x at x=0.
  //
  SOPHUS_FUNC Matrix<Scalar, DoF, num_parameters> Dx_this_mul_exp_x_at_0()
      const {
    Matrix<Scalar, DoF, num_parameters> J;
    Eigen::Quaternion<Scalar> const q = unit_quaternion();
    Scalar const c0 = q.w() * q.w();
    Scalar const c1 = q.x() * q.x();
    Scalar const c2 = q.y() * q.y();
    Scalar const c3 = -c2;
    Scalar const c4 = q.z() * q.z();
    Scalar const c5 = -c4;
    Scalar const c6 = Scalar(2) * q.w();
    Scalar const c7 = c6 * q.z();
    Scalar const c8 = Scalar(2) * q.x();
    Scalar const c9 = c8 * q.y();
    Scalar const c10 = c6 * q.y();
    Scalar const c11 = c8 * q.z();
    Scalar const c12 = c0 - c1;
    Scalar const c13 = c6 * q.x();
    Scalar const c14 = 2 * q.y() * q.z();
    Scalar const c15 = Scalar(0.5) * q.w();
    Scalar const c16 = Scalar(0.5) * q.z();
    Scalar const c17 = Scalar(0.5) * q.y();
    Scalar const c18 = -c17;
    Scalar const c19 = Scalar(0.5) * q.x();
    Scalar const c20 = -c19;
    Scalar const c21 = -c16;
    J(0, 0) = 0;
    J(0, 1) = 0;
    J(0, 2) = 0;
    J(0, 3) = 0;
    J(0, 4) = c0 + c1 + c3 + c5;
    J(0, 5) = c7 + c9;
    J(0, 6) = -c10 + c11;
    J(1, 0) = 0;
    J(1, 1) = 0;
    J(1, 2) = 0;
    J(1, 3) = 0;
    J(1, 4) = -c7 + c9;
    J(1, 5) = c12 + c2 + c5;
    J(1, 6) = c13 + c14;
    J(2, 0) = 0;
    J(2, 1) = 0;
    J(2, 2) = 0;
    J(2, 3) = 0;
    J(2, 4) = c10 + c11;
    J(2, 5) = -c13 + c14;
    J(2, 6) = c12 + c3 + c4;
    J(3, 0) = c15;
    J(3, 1) = c16;
    J(3, 2) = c18;
    J(3, 3) = c20;
    J(3, 4) = 0;
    J(3, 5) = 0;
    J(3, 6) = 0;
    J(4, 0) = c21;
    J(4, 1) = c15;
    J(4, 2) = c19;
    J(4, 3) = c18;
    J(4, 4) = 0;
    J(4, 5) = 0;
    J(4, 6) = 0;
    J(5, 0) = c17;
    J(5, 1) = c20;
    J(5, 2) = c15;
    J(5, 3) = c21;
    J(5, 4) = 0;
    J(5, 5) = 0;
    J(5, 6) = 0;
    return J;
  }

  // Returns group inverse.
  //
  SOPHUS_FUNC SE3<Scalar> inverse() const {
    SO3<Scalar> invR = so3().inverse();
    return SE3<Scalar>(invR, invR * (translation() * Scalar(-1)));
  }

  // Logarithmic map
  //
  // Computes the logarithm, the inverse of the group exponential which maps
  // element of the group (rigid body transformations) to elements of the
  // tangent space (twist).
  //
  // To be specific, this function computes ``vee(logmat(.))`` with
  // ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  // of SE(3).
  //
  SOPHUS_FUNC Tangent log() const {
    // For the derivation of the logarithm of SE(3), see
    // J. Gallier, D. Xu, "Computing exponentials of skew symmetric matrices and
    // logarithms of orthogonal matrices", IJRA 2002.
    // https://pdfs.semanticscholar.org/cfe3/e4b39de63c8cabd89bf3feff7f5449fc981d.pdf
    // (Sec. 6., pp. 8)
    using std::abs;
    using std::cos;
    using std::sin;
    Tangent upsilon_omega;
    auto omega_and_theta = so3().logAndTheta();
    Scalar theta = omega_and_theta.theta;
    upsilon_omega.template tail<3>() = omega_and_theta.tangent;
    Matrix3<Scalar> const Omega =
        SO3<Scalar>::hat(upsilon_omega.template tail<3>());

    if (abs(theta) < Constants<Scalar>::epsilon()) {
      Matrix3<Scalar> const V_inv = Matrix3<Scalar>::Identity() -
                                    Scalar(0.5) * Omega +
                                    Scalar(1. / 12.) * (Omega * Omega);

      upsilon_omega.template head<3>() = V_inv * translation();
    } else {
      Scalar const half_theta = Scalar(0.5) * theta;

      Matrix3<Scalar> const V_inv =
          (Matrix3<Scalar>::Identity() - Scalar(0.5) * Omega +
           (Scalar(1) -
            theta * cos(half_theta) / (Scalar(2) * sin(half_theta))) /
               (theta * theta) * (Omega * Omega));
      upsilon_omega.template head<3>() = V_inv * translation();
    }
    return upsilon_omega;
  }

  // It re-normalizes the SO3 element.
  //
  // Note: Because of the class invariant of SO3, there is typically no need to
  // call this function directly.
  //
  SOPHUS_FUNC void normalize() { so3().normalize(); }

  // Returns 4x4 matrix representation of the instance.
  //
  // It has the following form:
  //
  //   | R t |
  //   | o 1 |
  //
  // where ``R`` is a 3x3 rotation matrix, ``t`` a translation 3-vector and
  // ``o`` a 3-column vector of zeros.
  //
  SOPHUS_FUNC Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.template topLeftCorner<3, 4>() = matrix3x4();
    homogenious_matrix.row(3) =
        Matrix<Scalar, 1, 4>(Scalar(0), Scalar(0), Scalar(0), Scalar(1));
    return homogenious_matrix;
  }

  // Returns the significant first three rows of the matrix above.
  //
  SOPHUS_FUNC Matrix<Scalar, 3, 4> matrix3x4() const {
    Matrix<Scalar, 3, 4> matrix;
    matrix.template topLeftCorner<3, 3>() = rotationMatrix();
    matrix.col(3) = translation();
    return matrix;
  }

  // Assignment operator.
  //
  SOPHUS_FUNC SE3Base& operator=(SE3Base const& other) = default;

  // Assignment-like operator from OtherDerived.
  //
  template <class OtherDerived>
  SOPHUS_FUNC SE3Base<Derived>& operator=(SE3Base<OtherDerived> const& other) {
    so3() = other.so3();
    translation() = other.translation();
    return *this;
  }

  // Group multiplication, which is rotation concatenation.
  //
  SOPHUS_FUNC SE3<Scalar> operator*(SE3<Scalar> const& other) const {
    SE3<Scalar> result(*this);
    result *= other;
    return result;
  }

  // Group action on 3-points.
  //
  // This function rotates and translates a three dimensional point ``p`` by the
  // SE(3) element ``bar_T_foo = (bar_R_foo, t_bar)`` (= rigid body
  // transformation):
  //
  //   ``p_bar = bar_R_foo * p_foo + t_bar``.
  //
  SOPHUS_FUNC Point operator*(Point const& p) const {
    return so3() * p + translation();
  }

  // Group action on lines.
  //
  // This function rotates and translates a parametrized line
  // ``l(t) = o + t * d`` by the SE(3) element:
  //
  // Origin is transformed using SE(3) action
  // Direction is transformed using rotation part
  //
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), so3() * l.direction());
  }

  // In-place group multiplication.
  //
  SOPHUS_FUNC SE3Base<Derived>& operator*=(SE3<Scalar> const& other) {
    translation() += so3() * (other.translation());
    so3() *= other.so3();
    return *this;
  }

  // Returns rotation matrix.
  //
  SOPHUS_FUNC Matrix3<Scalar> rotationMatrix() const { return so3().matrix(); }

  // Mutator of SO3 group.
  //
  SOPHUS_FUNC SO3Type& so3() { return static_cast<Derived*>(this)->so3(); }

  // Accessor of SO3 group.
  //
  SOPHUS_FUNC SO3Type const& so3() const {
    return static_cast<const Derived*>(this)->so3();
  }

  // Takes in quaternion, and normalizes it.
  //
  // Precondition: The quaternion must not be close to zero.
  //
  SOPHUS_FUNC void setQuaternion(Eigen::Quaternion<Scalar> const& quat) {
    so3().setQuaternion(quat);
  }

  // Sets ``so3`` using ``rotation_matrix``.
  //
  // Precondition: ``R`` must be orthogonal and ``det(R)=1``.
  //
  SOPHUS_FUNC void setRotationMatrix(Matrix3<Scalar> const& R) {
    SOPHUS_ENSURE(isOrthogonal(R), "R is not orthogonal:\n %", R);
    SOPHUS_ENSURE(R.determinant() > 0, "det(R) is not positive: %",
                  R.determinant());
    so3().setQuaternion(Eigen::Quaternion<Scalar>(R));
  }

  // Returns internal parameters of SE(3).
  //
  // It returns (q.imag[0], q.imag[1], q.imag[2], q.real, t[0], t[1], t[2]),
  // with q being the unit quaternion, t the translation 3-vector.
  //
  SOPHUS_FUNC Sophus::Vector<Scalar, num_parameters> params() const {
    Sophus::Vector<Scalar, num_parameters> p;
    p << so3().params(), translation();
    return p;
  }

  // Mutator of translation vector.
  //
  SOPHUS_FUNC TranslationType& translation() {
    return static_cast<Derived*>(this)->translation();
  }

  // Accessor of translation vector
  //
  SOPHUS_FUNC TranslationType const& translation() const {
    return static_cast<Derived const*>(this)->translation();
  }

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC QuaternionType const& unit_quaternion() const {
    return this->so3().unit_quaternion();
  }
};

// SE3 default type - Constructors and default storage for SE3 Type.
template <class Scalar_, int Options>
class SE3 : public SE3Base<SE3<Scalar_, Options>> {
  using Base = SE3Base<SE3<Scalar_, Options>>;

 public:
  static int constexpr DoF = Base::DoF;
  static int constexpr num_parameters = Base::num_parameters;

  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using SO3Member = SO3<Scalar, Options>;
  using TranslationMember = Vector3<Scalar, Options>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor initialize rigid body motion to the identity.
  //
  SOPHUS_FUNC SE3() : translation_(Vector3<Scalar>::Zero()) {}

  // Copy constructor
  //
  SOPHUS_FUNC SE3(SE3 const& other) = default;

  // Copy-like constructor from OtherDerived.
  //
  template <class OtherDerived>
  SOPHUS_FUNC SE3(SE3Base<OtherDerived> const& other)
      : so3_(other.so3()), translation_(other.translation()) {
    static_assert(std::is_same<typename OtherDerived::Scalar, Scalar>::value,
                  "must be same Scalar type");
  }

  // Constructor from SO3 and translation vector
  //
  template <class OtherDerived, class D>
  SOPHUS_FUNC SE3(SO3Base<OtherDerived> const& so3,
                  Eigen::MatrixBase<D> const& translation)
      : so3_(so3), translation_(translation) {
    static_assert(std::is_same<typename OtherDerived::Scalar, Scalar>::value,
                  "must be same Scalar type");
    static_assert(std::is_same<typename D::Scalar, Scalar>::value,
                  "must be same Scalar type");
  }

  // Constructor from rotation matrix and translation vector
  //
  // Precondition: Rotation matrix needs to be orthogonal with determinant of 1.
  //
  SOPHUS_FUNC
  SE3(Matrix3<Scalar> const& rotation_matrix, Point const& translation)
      : so3_(rotation_matrix), translation_(translation) {}

  // Constructor from quaternion and translation vector.
  //
  // Precondition: quaternion must not be close to zero.
  //
  SOPHUS_FUNC SE3(Eigen::Quaternion<Scalar> const& quaternion,
                  Point const& translation)
      : so3_(quaternion), translation_(translation) {}

  // Constructor from 4x4 matrix
  //
  // Precondition: Rotation matrix needs to be orthogonal with determinant of 1.
  //               The last row must be (0, 0, 0, 1).
  //
  SOPHUS_FUNC explicit SE3(Matrix4<Scalar> const& T)
      : so3_(T.template topLeftCorner<3, 3>()),
        translation_(T.template block<3, 1>(0, 3)) {
    SOPHUS_ENSURE((T.row(3) - Matrix<Scalar, 1, 4>(Scalar(0), Scalar(0),
                                                   Scalar(0), Scalar(1)))
                          .squaredNorm() < Constants<Scalar>::epsilon(),
                  "Last row is not (0,0,0,1), but (%).", T.row(3));
  }

  // This provides unsafe read/write access to internal data. SO(3) is
  // represented by an Eigen::Quaternion (four parameters). When using direct
  // write access, the user needs to take care of that the quaternion stays
  // normalized.
  //
  SOPHUS_FUNC Scalar* data() {
    // so3_ and translation_ are laid out sequentially with no padding
    return so3_.data();
  }

  // Const version of data() above.
  //
  SOPHUS_FUNC Scalar const* data() const {
    // so3_ and translation_ are laid out sequentially with no padding
    return so3_.data();
  }

  // Accessor of SO3
  //
  SOPHUS_FUNC SO3Member& so3() { return so3_; }

  // Mutator of SO3
  //
  SOPHUS_FUNC SO3Member const& so3() const { return so3_; }

  // Mutator of translation vector
  //
  SOPHUS_FUNC TranslationMember& translation() { return translation_; }

  // Accessor of translation vector
  //
  SOPHUS_FUNC TranslationMember const& translation() const {
    return translation_;
  }

  // Returns derivative of exp(x) wrt. x.
  //
  SOPHUS_FUNC static Sophus::Matrix<Scalar, DoF, num_parameters> Dx_exp_x(
      Tangent const& upsilon_omega) {
    using std::pow;
    using std::sin;
    using std::cos;
    using std::sqrt;
    Sophus::Matrix<Scalar, DoF, num_parameters> J;
    Sophus::Vector<Scalar, 3> upsilon = upsilon_omega.template head<3>();
    Sophus::Vector<Scalar, 3> omega = upsilon_omega.template tail<3>();

    Scalar const c0 = omega[1] * omega[1];
    Scalar const c2 = omega[2] * omega[2];
    Scalar const c5 = omega[0] * omega[0];
    Scalar const c6 = c0 + c2 + c5;

    if (c6 < Constants<Scalar>::epsilon()) {
      Scalar const o(0);
      Scalar const h(0.5);
      Scalar const i(1);
      Scalar const ux = Scalar(0.5) * upsilon[0];
      Scalar const uy = Scalar(0.5) * upsilon[1];
      Scalar const uz = Scalar(0.5) * upsilon[2];

      // clang-format off
      J << o, o, o, o,  i,    o,   o,
           o, o, o, o,  o,    i,   o,
           o, o, o, o,  o,    o,   i,
           h, o, o, o,  o,  -uz,  uy,
           o, h, o, o,  uz,   o, -ux,
           o, o, h, o, -uy,  ux,   o;
      // clang-format on
      return J;
    }

    Scalar const c1 = -c0;
    Scalar const c3 = -c2;
    Scalar const c4 = c1 + c3;
    Scalar const c7 = pow(c6, Scalar(-3.0 / 2.0));
    Scalar const c8 = sqrt(c6);
    Scalar const c9 = sin(c8);
    Scalar const c10 = c8 - c9;
    Scalar const c11 = c10 * c7;
    Scalar const c12 = 1.0 / c6;
    Scalar const c13 = cos(c8);
    Scalar const c14 = -c13 + 1;
    Scalar const c15 = c12 * c14;
    Scalar const c16 = c15 * omega[2];
    Scalar const c17 = c11 * omega[0];
    Scalar const c18 = c17 * omega[1];
    Scalar const c19 = c15 * omega[1];
    Scalar const c20 = c17 * omega[2];
    Scalar const c21 = -c5;
    Scalar const c22 = c21 + c3;
    Scalar const c23 = c15 * omega[0];
    Scalar const c24 = omega[1] * omega[2];
    Scalar const c25 = c11 * c24;
    Scalar const c26 = c1 + c21;
    Scalar const c27 = 1.0 / c8;
    Scalar const c28 = 0.5 * c8;
    Scalar const c29 = sin(c28);
    Scalar const c30 = c27 * c29;
    Scalar const c31 = c29 * c7;
    Scalar const c32 = cos(c28);
    Scalar const c33 = 0.5 * c12 * c32;
    Scalar const c34 = c29 * c7 * omega[0];
    Scalar const c35 = 0.5 * c12 * c32 * omega[0];
    Scalar const c36 = -c34 * omega[1] + c35 * omega[1];
    Scalar const c37 = -c34 * omega[2] + c35 * omega[2];
    Scalar const c38 = c27 * omega[0];
    Scalar const c39 = 0.5 * c29;
    Scalar const c40 = pow(c6, -5.0L / 2.0L);
    Scalar const c41 = 3 * c10 * c40 * omega[0];
    Scalar const c42 = c4 * c7;
    Scalar const c43 = -c13 * c38 + c38;
    Scalar const c44 = c7 * c9 * omega[0];
    Scalar const c45 = c44 * omega[1];
    Scalar const c46 = pow(c6, -2);
    Scalar const c47 = 2 * c14 * c46 * omega[0];
    Scalar const c48 = c47 * omega[1];
    Scalar const c49 = c11 * omega[2];
    Scalar const c50 = c45 - c48 + c49;
    Scalar const c51 = 3 * c10 * c40 * c5;
    Scalar const c52 = c7 * omega[0] * omega[2];
    Scalar const c53 = c43 * c52 - c51 * omega[2];
    Scalar const c54 = c7 * omega[0] * omega[1];
    Scalar const c55 = c43 * c54 - c51 * omega[1];
    Scalar const c56 = c44 * omega[2];
    Scalar const c57 = c47 * omega[2];
    Scalar const c58 = c11 * omega[1];
    Scalar const c59 = -c56 + c57 + c58;
    Scalar const c60 = -2 * c17;
    Scalar const c61 = c22 * c7;
    Scalar const c62 = -c24 * c41;
    Scalar const c63 = -c15 + c62;
    Scalar const c64 = c7 * c9;
    Scalar const c65 = c5 * c64;
    Scalar const c66 = 2 * c14 * c46;
    Scalar const c67 = c5 * c66;
    Scalar const c68 = c7 * omega[1] * omega[2];
    Scalar const c69 = c43 * c68;
    Scalar const c70 = c56 - c57 + c58;
    Scalar const c71 = c26 * c7;
    Scalar const c72 = c15 + c62;
    Scalar const c73 = -c45 + c48 + c49;
    Scalar const c74 = -c24 * c31 + c24 * c33;
    Scalar const c75 = c27 * omega[1];
    Scalar const c76 = -2 * c58;
    Scalar const c77 = 3 * c10 * c40 * omega[1];
    Scalar const c78 = -c13 * c75 + c75;
    Scalar const c79 = c0 * c64;
    Scalar const c80 = c0 * c66;
    Scalar const c81 = c52 * c78;
    Scalar const c82 = -c0 * c41 + c54 * c78;
    Scalar const c83 = c24 * c64;
    Scalar const c84 = c24 * c66;
    Scalar const c85 = c17 - c83 + c84;
    Scalar const c86 = c17 + c83 - c84;
    Scalar const c87 = 3 * c10 * c40 * omega[2];
    Scalar const c88 = -c0 * c87 + c68 * c78;
    Scalar const c89 = c27 * omega[2];
    Scalar const c90 = -2 * c49;
    Scalar const c91 = -c13 * c89 + c89;
    Scalar const c92 = c2 * c64;
    Scalar const c93 = c2 * c66;
    Scalar const c94 = c54 * c91;
    Scalar const c95 = -c2 * c41 + c52 * c91;
    Scalar const c96 = -c2 * c77 + c68 * c91;
    J(0, 0) = 0;
    J(0, 1) = 0;
    J(0, 2) = 0;
    J(0, 3) = 0;
    J(0, 4) = c11 * c4 + 1;
    J(0, 5) = c16 + c18;
    J(0, 6) = -c19 + c20;
    J(1, 0) = 0;
    J(1, 1) = 0;
    J(1, 2) = 0;
    J(1, 3) = 0;
    J(1, 4) = -c16 + c18;
    J(1, 5) = c11 * c22 + 1;
    J(1, 6) = c23 + c25;
    J(2, 0) = 0;
    J(2, 1) = 0;
    J(2, 2) = 0;
    J(2, 3) = 0;
    J(2, 4) = c19 + c20;
    J(2, 5) = -c23 + c25;
    J(2, 6) = c11 * c26 + 1;
    J(3, 0) = c30 - c31 * c5 + c33 * c5;
    J(3, 1) = c36;
    J(3, 2) = c37;
    J(3, 3) = -c38 * c39;
    J(3, 4) = upsilon[0] * (-c4 * c41 + c42 * c43) + upsilon[1] * (c55 + c59) +
              upsilon[2] * (c50 + c53);
    J(3, 5) = upsilon[0] * (c55 + c70) +
              upsilon[1] * (-c22 * c41 + c43 * c61 + c60) +
              upsilon[2] * (c63 - c65 + c67 + c69);
    J(3, 6) = upsilon[0] * (c53 + c73) + upsilon[1] * (c65 - c67 + c69 + c72) +
              upsilon[2] * (-c26 * c41 + c43 * c71 + c60);
    J(4, 0) = c36;
    J(4, 1) = -c0 * c31 + c0 * c33 + c30;
    J(4, 2) = c74;
    J(4, 3) = -c39 * c75;
    J(4, 4) = upsilon[0] * (-c4 * c77 + c42 * c78 + c76) +
              upsilon[1] * (c82 + c85) + upsilon[2] * (c72 + c79 - c80 + c81);
    J(4, 5) = upsilon[0] * (c82 + c86) + upsilon[1] * (-c22 * c77 + c61 * c78) +
              upsilon[2] * (c73 + c88);
    J(4, 6) = upsilon[0] * (c63 - c79 + c80 + c81) + upsilon[1] * (c50 + c88) +
              upsilon[2] * (-c26 * c77 + c71 * c78 + c76);
    J(5, 0) = c37;
    J(5, 1) = c74;
    J(5, 2) = -c2 * c31 + c2 * c33 + c30;
    J(5, 3) = -c39 * c89;
    J(5, 4) = upsilon[0] * (-c4 * c87 + c42 * c91 + c90) +
              upsilon[1] * (c63 - c92 + c93 + c94) + upsilon[2] * (c86 + c95);
    J(5, 5) = upsilon[0] * (c72 + c92 - c93 + c94) +
              upsilon[1] * (-c22 * c87 + c61 * c91 + c90) +
              upsilon[2] * (c59 + c96);
    J(5, 6) = upsilon[0] * (c85 + c95) + upsilon[1] * (c70 + c96) +
              upsilon[2] * (-c26 * c87 + c71 * c91);

    return J;
  }

  // Returns derivative of exp(x) wrt. x_i at x=0.
  //
  SOPHUS_FUNC static Sophus::Matrix<Scalar, DoF, num_parameters>
  Dx_exp_x_at_0() {
    Sophus::Matrix<Scalar, DoF, num_parameters> J;
    Scalar const o(0);
    Scalar const h(0.5);
    Scalar const i(1);

    // clang-format off
    J << o, o, o, o, i, o, o,
         o, o, o, o, o, i, o,
         o, o, o, o, o, o, i,
         h, o, o, o, o, o, o,
         o, h, o, o, o, o, o,
         o, o, h, o, o, o, o;
    // clang-format on
    return J;
  }

  // Returns derivative of exp(x).matrix() wrt. x_i at x=0.
  //
  SOPHUS_FUNC static Transformation Dxi_exp_x_matrix_at_0(int i) {
    return generator(i);
  }

  // Group exponential
  //
  // This functions takes in an element of tangent space (= twist ``a``) and
  // returns the corresponding element of the group SE(3).
  //
  // The first three components of ``a`` represent the translational part
  // ``upsilon`` in the tangent space of SE(3), while the last three components
  // of ``a`` represents the rotation vector ``omega``.
  // To be more specific, this function computes ``expmat(hat(a))`` with
  // ``expmat(.)`` being the matrix exponential and ``hat(.)`` the hat-operator
  // of SE(3), see below.
  //
  SOPHUS_FUNC static SE3<Scalar> exp(Tangent const& a) {
    using std::cos;
    using std::sin;
    Vector3<Scalar> const omega = a.template tail<3>();

    Scalar theta;
    SO3<Scalar> const so3 = SO3<Scalar>::expAndTheta(omega, &theta);
    Matrix3<Scalar> const Omega = SO3<Scalar>::hat(omega);
    Matrix3<Scalar> const Omega_sq = Omega * Omega;
    Matrix3<Scalar> V;

    if (theta < Constants<Scalar>::epsilon()) {
      V = so3.matrix();
      // Note: That is an accurate expansion!
    } else {
      Scalar theta_sq = theta * theta;
      V = (Matrix3<Scalar>::Identity() +
           (Scalar(1) - cos(theta)) / (theta_sq)*Omega +
           (theta - sin(theta)) / (theta_sq * theta) * Omega_sq);
    }
    return SE3<Scalar>(so3, V * a.template head<3>());
  }

  // Returns closest SE3 given arbirary 4x4 matrix.
  //
  SOPHUS_FUNC static SE3 fitToSE3(Matrix4<Scalar> const& T) {
    return SE3(SO3<Scalar>::fitToSO3(T.template block<3, 3>(0, 0)),
               T.template block<3, 1>(0, 3));
  }

  // Returns the ith infinitesimal generators of SE(3).
  //
  // The infinitesimal generators of SE(3) are:
  //
  //         |  0  0  0  1 |
  //   G_0 = |  0  0  0  0 |
  //         |  0  0  0  0 |
  //         |  0  0  0  0 |
  //
  //         |  0  0  0  0 |
  //   G_1 = |  0  0  0  1 |
  //         |  0  0  0  0 |
  //         |  0  0  0  0 |
  //
  //         |  0  0  0  0 |
  //   G_2 = |  0  0  0  0 |
  //         |  0  0  0  1 |
  //         |  0  0  0  0 |
  //
  //         |  0  0  0  0 |
  //   G_3 = |  0  0 -1  0 |
  //         |  0  1  0  0 |
  //         |  0  0  0  0 |
  //
  //         |  0  0  1  0 |
  //   G_4 = |  0  0  0  0 |
  //         | -1  0  0  0 |
  //         |  0  0  0  0 |
  //
  //         |  0 -1  0  0 |
  //   G_5 = |  1  0  0  0 |
  //         |  0  0  0  0 |
  //         |  0  0  0  0 |
  //
  // Precondition: ``i`` must be in [0, 5].
  //
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 && i <= 5, "i should be in range [0,5].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  // hat-operator
  //
  // It takes in the 6-vector representation (= twist) and returns the
  // corresponding matrix representation of Lie algebra element.
  //
  // Formally, the ``hat()`` operator of SE(3) is defined as
  //
  //   ``hat(.): R^6 -> R^{4x4},  hat(a) = sum_i a_i * G_i``  (for i=0,...,5)
  //
  // with ``G_i`` being the ith infinitesimal generator of SE(3).
  //
  SOPHUS_FUNC static Transformation hat(Tangent const& a) {
    Transformation Omega;
    Omega.setZero();
    Omega.template topLeftCorner<3, 3>() =
        SO3<Scalar>::hat(a.template tail<3>());
    Omega.col(3).template head<3>() = a.template head<3>();
    return Omega;
  }

  // Lie bracket
  //
  // It computes the Lie bracket of SE(3). To be more specific, it computes
  //
  //   ``[omega_1, omega_2]_se3 := vee([hat(omega_1), hat(omega_2)])``
  //
  // with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.) the
  // hat-operator and ``vee(.)`` the vee-operator of SE(3).
  //
  SOPHUS_FUNC static Tangent lieBracket(Tangent const& a, Tangent const& b) {
    Vector3<Scalar> const upsilon1 = a.template head<3>();
    Vector3<Scalar> const upsilon2 = b.template head<3>();
    Vector3<Scalar> const omega1 = a.template tail<3>();
    Vector3<Scalar> const omega2 = b.template tail<3>();

    Tangent res;
    res.template head<3>() = omega1.cross(upsilon2) + upsilon1.cross(omega2);
    res.template tail<3>() = omega1.cross(omega2);

    return res;
  }

  // Construct x-axis rotation.
  //
  static SOPHUS_FUNC SE3 rotX(Scalar const& x) {
    return SE3(SO3<Scalar>::rotX(x), Sophus::Vector3<Scalar>::Zero());
  }

  // Construct y-axis rotation.
  //
  static SOPHUS_FUNC SE3 rotY(Scalar const& y) {
    return SE3(SO3<Scalar>::rotY(y), Sophus::Vector3<Scalar>::Zero());
  }

  // Construct z-axis rotation.
  //
  static SOPHUS_FUNC SE3 rotZ(Scalar const& z) {
    return SE3(SO3<Scalar>::rotZ(z), Sophus::Vector3<Scalar>::Zero());
  }

  // Draw uniform sample from SE(3) manifold.
  //
  // Translations are drawn component-wise from the range [-1, 1].
  //
  template <class UniformRandomBitGenerator>
  static SE3 sampleUniform(UniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    return SE3(SO3<Scalar>::sampleUniform(generator),
               Vector3<Scalar>(uniform(generator), uniform(generator),
                               uniform(generator)));
  }

  // Construct a translation only SE3 instance.
  //
  template <class T0, class T1, class T2>
  static SOPHUS_FUNC SE3 trans(T0 const& x, T1 const& y, T2 const& z) {
    return SE3(SO3<Scalar>(), Vector3<Scalar>(x, y, z));
  }

  // Construct x-axis translation.
  //
  static SOPHUS_FUNC SE3 transX(Scalar const& x) {
    return SE3::trans(x, Scalar(0), Scalar(0));
  }

  // Construct y-axis translation.
  //
  static SOPHUS_FUNC SE3 transY(Scalar const& y) {
    return SE3::trans(Scalar(0), y, Scalar(0));
  }

  // Construct z-axis translation.
  //
  static SOPHUS_FUNC SE3 transZ(Scalar const& z) {
    return SE3::trans(Scalar(0), Scalar(0), z);
  }

  // vee-operator
  //
  // It takes 4x4-matrix representation ``Omega`` and maps it to the
  // corresponding 6-vector representation of Lie algebra.
  //
  // This is the inverse of the hat-operator, see above.
  //
  // Precondition: ``Omega`` must have the following structure:
  //
  //                |  0 -f  e  a |
  //                |  f  0 -d  b |
  //                | -e  d  0  c |
  //                |  0  0  0  0 | .
  //
  SOPHUS_FUNC static Tangent vee(Transformation const& Omega) {
    Tangent upsilon_omega;
    upsilon_omega.template head<3>() = Omega.col(3).template head<3>();
    upsilon_omega.template tail<3>() =
        SO3<Scalar>::vee(Omega.template topLeftCorner<3, 3>());
    return upsilon_omega;
  }

 protected:
  SO3Member so3_;
  TranslationMember translation_;
};

}  // namespace Sophus

namespace Eigen {

// Specialization of Eigen::Map for ``SE3``.
//
// Allows us to wrap SE3 objects around POD array.
template <class Scalar_, int Options>
class Map<Sophus::SE3<Scalar_>, Options>
    : public Sophus::SE3Base<Map<Sophus::SE3<Scalar_>, Options>> {
  using Base = Sophus::SE3Base<Map<Sophus::SE3<Scalar_>, Options>>;

 public:
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  // LCOV_EXCL_START
  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  // LCOV_EXCL_END

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar* coeffs)
      : so3_(coeffs),
        translation_(coeffs + Sophus::SO3<Scalar>::num_parameters) {}

  // Mutator of SO3
  //
  SOPHUS_FUNC Map<Sophus::SO3<Scalar>, Options>& so3() { return so3_; }

  // Accessor of SO3
  //
  SOPHUS_FUNC Map<Sophus::SO3<Scalar>, Options> const& so3() const {
    return so3_;
  }

  // Mutator of translation vector
  //
  SOPHUS_FUNC Map<Sophus::Vector3<Scalar>>& translation() {
    return translation_;
  }

  // Accessor of translation vector
  //
  SOPHUS_FUNC Map<Sophus::Vector3<Scalar>> const& translation() const {
    return translation_;
  }

 protected:
  Map<Sophus::SO3<Scalar>, Options> so3_;
  Map<Sophus::Vector3<Scalar>, Options> translation_;
};

// Specialization of Eigen::Map for ``SE3 const``.
//
// Allows us to wrap SE3 objects around POD array.
template <class Scalar_, int Options>
class Map<Sophus::SE3<Scalar_> const, Options>
    : public Sophus::SE3Base<Map<Sophus::SE3<Scalar_> const, Options>> {
  using Base = Sophus::SE3Base<Map<Sophus::SE3<Scalar_> const, Options>>;

 public:
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar const* coeffs)
      : so3_(coeffs),
        translation_(coeffs + Sophus::SO3<Scalar>::num_parameters) {}

  // Accessor of SO3
  //
  SOPHUS_FUNC Map<Sophus::SO3<Scalar> const, Options> const& so3() const {
    return so3_;
  }

  // Accessor of translation vector
  //
  SOPHUS_FUNC Map<Sophus::Vector3<Scalar> const, Options> const& translation()
      const {
    return translation_;
  }

 protected:
  Map<Sophus::SO3<Scalar> const, Options> const so3_;
  Map<Sophus::Vector3<Scalar> const, Options> const translation_;
};
}

#endif
