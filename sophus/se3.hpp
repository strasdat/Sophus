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

  // Returns ``*this`` times the ith generator of internal representation.
  //
  SOPHUS_FUNC Vector<Scalar, num_parameters> internalMultiplyByGenerator(
      int i) const {
    Vector<Scalar, num_parameters> res;

    Eigen::Quaternion<Scalar> internal_gen_q;
    Vector<Scalar, 3> internal_gen_t;

    internalGenerator(i, &internal_gen_q, &internal_gen_t);

    res.template head<4>() =
        (so3().unit_quaternion() * internal_gen_q).coeffs();
    res.template tail<3>() = so3().unit_quaternion() * internal_gen_t;
    return res;
  }

  // Returns Jacobian of generator of internal SU(2) representation.
  //
  SOPHUS_FUNC Matrix<Scalar, num_parameters, DoF> internalJacobian() const {
    Matrix<Scalar, num_parameters, DoF> J;
    for (int i = 0; i < DoF; ++i) {
      J.col(i) = internalMultiplyByGenerator(i);
    }
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
  // Returns tangent space representation (= twist) of the instance.
  //
  SOPHUS_FUNC Tangent log() const { return log(*this); }

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

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  // Derivative of Lie bracket with respect to first element.
  //
  // This function returns ``D_a [a, b]`` with ``D_a`` being the
  // differential operator with respect to ``a``, ``[a, b]`` being the lie
  // bracket of the Lie algebra se(3).
  // See ``lieBracket()`` below.
  //
  SOPHUS_FUNC static Adjoint d_lieBracketab_by_d_a(Tangent const& b) {
    Adjoint res;
    res.setZero();

    Vector3<Scalar> const upsilon2 = b.template head<3>();
    Vector3<Scalar> const omega2 = b.template tail<3>();

    res.template topLeftCorner<3, 3>() = -SO3<Scalar>::hat(omega2);
    res.template topRightCorner<3, 3>() = -SO3<Scalar>::hat(upsilon2);
    res.template bottomRightCorner<3, 3>() = -SO3<Scalar>::hat(omega2);
    return res;
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

  // Returns the ith generator of internal representation.
  //
  // Precondition: ``i`` must be in [0, 5].
  //
  SOPHUS_FUNC static void internalGenerator(
      int i, Eigen::Quaternion<Scalar>* internal_gen_q,
      Vector3<Scalar>* internal_gen_t) {
    SOPHUS_ENSURE(i >= 0 && i <= 5, "i should be in range [0,5]");
    SOPHUS_ENSURE(internal_gen_q != NULL,
                  "internal_gen_q must not be the null pointer");
    SOPHUS_ENSURE(internal_gen_t != NULL,
                  "internal_gen_t must not be the null pointer");

    internal_gen_q->coeffs().setZero();
    internal_gen_t->setZero();
    if (i < 3) {
      (*internal_gen_t)[i] = Scalar(1);
    } else {
      SO3<Scalar>::internalGenerator(i - 3, internal_gen_q);
      ;
    }
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
  SOPHUS_FUNC static Tangent log(SE3<Scalar> const& se3) {
    // For the derivation of the logarithm of SE(3), see
    // J. Gallier, D. Xu, "Computing exponentials of skew symmetric matrices and
    // logarithms of orthogonal matrices", IJRA 2002.
    // https://pdfs.semanticscholar.org/cfe3/e4b39de63c8cabd89bf3feff7f5449fc981d.pdf
    // (Sec. 6., pp. 8)
    using std::abs;
    using std::cos;
    using std::sin;
    Tangent upsilon_omega;
    Scalar theta;
    upsilon_omega.template tail<3>() =
        SO3<Scalar>::logAndTheta(se3.so3(), &theta);
    Matrix3<Scalar> const Omega =
        SO3<Scalar>::hat(upsilon_omega.template tail<3>());

    if (abs(theta) < Constants<Scalar>::epsilon()) {
      Matrix3<Scalar> const V_inv = Matrix3<Scalar>::Identity() -
                                    Scalar(0.5) * Omega +
                                    Scalar(1. / 12.) * (Omega * Omega);

      upsilon_omega.template head<3>() = V_inv * se3.translation();
    } else {
      Scalar const half_theta = Scalar(0.5) * theta;

      Matrix3<Scalar> const V_inv =
          (Matrix3<Scalar>::Identity() - Scalar(0.5) * Omega +
           (Scalar(1) -
            theta * cos(half_theta) / (Scalar(2) * sin(half_theta))) /
               (theta * theta) * (Omega * Omega));
      upsilon_omega.template head<3>() = V_inv * se3.translation();
    }
    return upsilon_omega;
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
};

// SE3 default type - Constructors and default storage for SE3 Type.
template <class Scalar_, int Options>
class SE3 : public SE3Base<SE3<Scalar_, Options>> {
  using Base = SE3Base<SE3<Scalar_, Options>>;

 public:
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
    SOPHUS_ENSURE((T.row(3) - Matrix<Scalar, 1, 4>(0, 0, 0, 1)).squaredNorm() <
                      Constants<Scalar>::epsilon(),
                  "Last row is not (0,0,0,1), but (%).", T.row(3));
  }

  // Returns closest SE3 given arbirary 4x4 matrix.
  //
  SOPHUS_FUNC static SE3 fitToSE3(Matrix4<Scalar> const& T) {
    return SE3(SO3<Scalar>::fitToSO3(T.template block<3, 3>(0, 0)),
               T.template block<3, 1>(0, 3));
  }

  // Construct a translation only SE3 instance.
  //
  template <class T0, class T1, class T2>
  static SOPHUS_FUNC SE3 trans(T0 const& x, T1 const& y, T2 const& z) {
    return SE3(SO3<Scalar>(), Vector3<Scalar>(x, y, z));
  }

  // Contruct x-axis translation.
  //
  static SOPHUS_FUNC SE3 transX(Scalar const& x) {
    return SE3::trans(x, Scalar(0), Scalar(0));
  }

  // Contruct y-axis translation.
  //
  static SOPHUS_FUNC SE3 transY(Scalar const& y) {
    return SE3::trans(Scalar(0), y, Scalar(0));
  }

  // Contruct z-axis translation.
  //
  static SOPHUS_FUNC SE3 transZ(Scalar const& z) {
    return SE3::trans(Scalar(0), Scalar(0), z);
  }

  // Contruct x-axis rotation.
  //
  static SOPHUS_FUNC SE3 rotX(Scalar const& x) {
    return SE3(SO3<Scalar>::rotX(x), Sophus::Vector3<Scalar>::Zero());
  }

  // Contruct y-axis rotation.
  //
  static SOPHUS_FUNC SE3 rotY(Scalar const& y) {
    return SE3(SO3<Scalar>::rotY(y), Sophus::Vector3<Scalar>::Zero());
  }

  // Contruct z-axis rotation.
  //
  static SOPHUS_FUNC SE3 rotZ(Scalar const& z) {
    return SE3(SO3<Scalar>::rotZ(z), Sophus::Vector3<Scalar>::Zero());
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

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
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

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
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
