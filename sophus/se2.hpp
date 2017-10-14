#ifndef SOPHUS_SE2_HPP
#define SOPHUS_SE2_HPP

#include "so2.hpp"

namespace Sophus {
template <class Scalar_, int Options = 0>
class SE2;
using SE2d = SE2<double>;
using SE2f = SE2<float>;
}  // namespace Sophus

namespace Eigen {
namespace internal {

template <class Scalar_, int Options>
struct traits<Sophus::SE2<Scalar_, Options>> {
  using Scalar = Scalar_;
  using TranslationType = Sophus::Vector2<Scalar, Options>;
  using SO2Type = Sophus::SO2<Scalar, Options>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::SE2<Scalar_>, Options>>
    : traits<Sophus::SE2<Scalar_, Options>> {
  using Scalar = Scalar_;
  using TranslationType = Map<Sophus::Vector2<Scalar>, Options>;
  using SO2Type = Map<Sophus::SO2<Scalar>, Options>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::SE2<Scalar_> const, Options>>
    : traits<Sophus::SE2<Scalar_, Options> const> {
  using Scalar = Scalar_;
  using TranslationType = Map<Sophus::Vector2<Scalar> const, Options>;
  using SO2Type = Map<Sophus::SO2<Scalar> const, Options>;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {

// SE2 base type - implements SE2 class but is storage agnostic.
//
// SE(2) is the group of rotations  and translation in 2d. It is the semi-direct
// product of SO(2) and the 2d Euclidean vector space.  The class is represented
// using a composition of SO2Group  for rotation and a 2-vector for translation.
//
// SE(2) is neither compact, nor a commutative group.
//
// See SO2Group for more details of the rotation representation in 2d.
//
template <class Derived>
class SE2Base {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using TranslationType =
      typename Eigen::internal::traits<Derived>::TranslationType;
  using SO2Type = typename Eigen::internal::traits<Derived>::SO2Type;

  // Degrees of freedom of manifold, number of dimensions in tangent space
  // (two for translation, three for rotation).
  static int constexpr DoF = 3;
  // Number of internal parameters used (tuple for complex, two for
  // translation).
  static int constexpr num_parameters = 4;
  // Group transformations are 3x3 matrices.
  static int constexpr N = 3;
  using Transformation = Matrix<Scalar, N, N>;
  using Point = Vector2<Scalar>;
  using Line = ParametrizedLine2<Scalar>;
  using Tangent = Vector<Scalar, DoF>;
  using Adjoint = Matrix<Scalar, DoF, DoF>;

  // Adjoint transformation
  //
  // This function return the adjoint transformation ``Ad`` of the group
  // element ``A`` such that for all ``x`` it holds that
  // ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  //
  SOPHUS_FUNC Adjoint Adj() const {
    Matrix<Scalar, 2, 2> const& R = so2().matrix();
    Transformation res;
    res.setIdentity();
    res.template topLeftCorner<2, 2>() = R;
    res(0, 2) = translation()[1];
    res(1, 2) = -translation()[0];
    return res;
  }

  // Returns copy of instance casted to NewScalarType.
  //
  template <class NewScalarType>
  SOPHUS_FUNC SE2<NewScalarType> cast() const {
    return SE2<NewScalarType>(so2().template cast<NewScalarType>(),
                              translation().template cast<NewScalarType>());
  }

  // Returns group inverse.
  //
  SOPHUS_FUNC SE2<Scalar> inverse() const {
    SO2<Scalar> const invR = so2().inverse();
    return SE2<Scalar>(invR, invR * (translation() * Scalar(-1)));
  }

  // Logarithmic map
  //
  // Returns tangent space representation (= twist) of the instance.
  //
  SOPHUS_FUNC Tangent log() const { return log(*this); }

  /**
   * \brief Normalize SO2 element
   *
   * It re-normalizes the SO2 element.
   */
  SOPHUS_FUNC void normalize() { so2().normalize(); }

  // Returns 3x3 matrix representation of the instance.
  //
  // It has the following form:
  //
  //   | R t |
  //   | o 1 |
  //
  // where ``R`` is a 2x2 rotation matrix, ``t`` a translation 2-vector and
  // ``o`` a 2-column vector of zeros.
  //
  SOPHUS_FUNC Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.template topLeftCorner<2, 3>() = matrix2x3();
    homogenious_matrix.row(2) =
        Matrix<Scalar, 1, 3>(Scalar(0), Scalar(0), Scalar(1));
    return homogenious_matrix;
  }

  // Returns the significant first two rows of the matrix above.
  //
  SOPHUS_FUNC Matrix<Scalar, 2, 3> matrix2x3() const {
    Matrix<Scalar, 2, 3> matrix;
    matrix.template topLeftCorner<2, 2>() = rotationMatrix();
    matrix.col(2) = translation();
    return matrix;
  }

  // Assignment operator.
  //
  template <class OtherDerived>
  SOPHUS_FUNC SE2Base<Derived>& operator=(SE2Base<OtherDerived> const& other) {
    so2() = other.so2();
    translation() = other.translation();
    return *this;
  }

  // Group multiplication, which is rotation concatenation.
  //
  SOPHUS_FUNC SE2<Scalar> operator*(SE2<Scalar> const& other) const {
    SE2<Scalar> result(*this);
    result *= other;
    return result;
  }

  // Group action on 2-points.
  //
  // This function rotates and translates a two dimensional point ``p`` by the
  // SE(2) element ``bar_T_foo = (bar_R_foo, t_bar)`` (= rigid body
  // transformation):
  //
  //   ``p_bar = bar_R_foo * p_foo + t_bar``.
  //
  SOPHUS_FUNC Point operator*(Point const& p) const {
    return so2() * p + translation();
  }

  // Group action on lines.
  //
  // This function rotates and translates a parametrized line
  // ``l(t) = o + t * d`` by the SE(2) element:
  //
  // Origin ``o`` is rotated and translated using SE(2) action
  // Direction ``d`` is rotated using SO(2) action
  //
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), so2() * l.direction());
  }

  // In-place group multiplication.
  //
  SOPHUS_FUNC SE2Base<Derived>& operator*=(SE2<Scalar> const& other) {
    translation() += so2() * (other.translation());
    so2() *= other.so2();
    return *this;
  }

  // Returns rotation matrix.
  //
  SOPHUS_FUNC Matrix<Scalar, 2, 2> rotationMatrix() const {
    return so2().matrix();
  }

  // Takes in complex number, and normalizes it.
  //
  // Precondition: The complex number must not be close to zero.
  //
  SOPHUS_FUNC void setComplex(Sophus::Vector2<Scalar> const& complex) {
    return so2().setComplex(complex);
  }

  // Sets ``so3`` using ``rotation_matrix``.
  //
  // Precondition: ``R`` must be orthogonal and ``det(R)=1``.
  //
  SOPHUS_FUNC void setRotationMatrix(Matrix<Scalar, 2, 2> const& R) {
    SOPHUS_ENSURE(isOrthogonal(R), "R is not orthogonal:\n %", R);
    SOPHUS_ENSURE(R.determinant() > 0, "det(R) is not positive: %",
                  R.determinant());
    so2().setComplex(Scalar(0.5) * (R(0, 0) + R(1, 1)),
                     Scalar(0.5) * (R(1, 0) - R(0, 1)));
  }

  // Mutator of SO3 group.
  //
  SOPHUS_FUNC
  SO2Type& so2() { return static_cast<Derived*>(this)->so2(); }

  // Accessor of SO3 group.
  //
  SOPHUS_FUNC
  SO2Type const& so2() const {
    return static_cast<Derived const*>(this)->so2();
  }

  // Mutator of translation vector.
  //
  SOPHUS_FUNC
  TranslationType& translation() {
    return static_cast<Derived*>(this)->translation();
  }

  // Accessor of translation vector
  //
  SOPHUS_FUNC
  TranslationType const& translation() const {
    return static_cast<Derived const*>(this)->translation();
  }

  // Accessor of unit complex number.
  //
  SOPHUS_FUNC
  typename Eigen::internal::traits<Derived>::SO2Type::ComplexT const&
  unit_complex() const {
    return so2().unit_complex();
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  // Derivative of Lie bracket with respect to first element.
  //
  // This function returns ``D_a [a, b]`` with ``D_a`` being the
  // differential operator with respect to ``a``, ``[a, b]`` being the lie
  // bracket of the Lie algebra se3.
  // See ``lieBracket()`` below.
  //
  SOPHUS_FUNC static Transformation d_lieBracketab_by_d_a(Tangent const& b) {
    static Scalar const zero = Scalar(0);
    Vector2<Scalar> upsilon2 = b.template head<2>();
    Scalar theta2 = b[2];

    Transformation res;
    res << zero, theta2, -upsilon2[1], -theta2, zero, upsilon2[0], zero, zero,
        zero;
    return res;
  }

  // Group exponential
  //
  // This functions takes in an element of tangent space (= twist ``a``) and
  // returns the corresponding element of the group SE(2).
  //
  // The first two components of ``a`` represent the translational part
  // ``upsilon`` in the tangent space of SE(2), while the last three components
  // of ``a`` represents the rotation vector ``omega``.
  // To be more specific, this function computes ``expmat(hat(a))`` with
  // ``expmat(.)`` being the matrix exponential and ``hat(.)`` the hat-operator
  // of SE(2), see below.
  //
  SOPHUS_FUNC static SE2<Scalar> exp(Tangent const& a) {
    Scalar theta = a[2];
    SO2<Scalar> so2 = SO2<Scalar>::exp(theta);
    Scalar sin_theta_by_theta;
    Scalar one_minus_cos_theta_by_theta;

    if (std::abs(theta) < Constants<Scalar>::epsilon()) {
      Scalar theta_sq = theta * theta;
      sin_theta_by_theta = Scalar(1.) - Scalar(1. / 6.) * theta_sq;
      one_minus_cos_theta_by_theta =
          Scalar(0.5) * theta - Scalar(1. / 24.) * theta * theta_sq;
    } else {
      sin_theta_by_theta = so2.unit_complex().y() / theta;
      one_minus_cos_theta_by_theta =
          (Scalar(1.) - so2.unit_complex().x()) / theta;
    }
    Vector2<Scalar> trans(
        sin_theta_by_theta * a[0] - one_minus_cos_theta_by_theta * a[1],
        one_minus_cos_theta_by_theta * a[0] + sin_theta_by_theta * a[1]);
    return SE2<Scalar>(so2, trans);
  }

  // Returns the ith infinitesimal generators of SE(2).
  //
  // The infinitesimal generators of SE(2) are:
  //
  //         |  0  0  1 |
  //   G_0 = |  0  0  0 |
  //         |  0  0  0 |
  //
  //         |  0  0  0 |
  //   G_1 = |  0  0  1 |
  //         |  0  0  0 |
  //
  //         |  0 -1  0 |
  //   G_2 = |  1  0  0 |
  //         |  0  0  0 |
  // Precondition: ``i`` must be in 0, 1 or 2.
  //
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 || i <= 2, "i should be in range [0,2].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  // hat-operator
  //
  // It takes in the 3-vector representation (= twist) and returns the
  // corresponding matrix representation of Lie algebra element.
  //
  // Formally, the ``hat()`` operator of SE(3) is defined as
  //
  //   ``hat(.): R^3 -> R^{3x33},  hat(a) = sum_i a_i * G_i``  (for i=0,1,2)
  //
  // with ``G_i`` being the ith infinitesimal generator of SE(2).
  //
  SOPHUS_FUNC static Transformation hat(Tangent const& a) {
    Transformation Omega;
    Omega.setZero();
    Omega.template topLeftCorner<2, 2>() = SO2<Scalar>::hat(a[2]);
    Omega.col(2).template head<2>() = a.template head<2>();
    return Omega;
  }

  // Lie bracket
  //
  // It computes the Lie bracket of SE(2). To be more specific, it computes
  //
  //   ``[omega_1, omega_2]_se2 := vee([hat(omega_1), hat(omega_2)])``
  //
  // with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.) the
  // hat-operator and ``vee(.)`` the vee-operator of SE(2).
  //
  SOPHUS_FUNC static Tangent lieBracket(Tangent const& a, Tangent const& b) {
    Vector2<Scalar> upsilon1 = a.template head<2>();
    Vector2<Scalar> upsilon2 = b.template head<2>();
    Scalar theta1 = a[2];
    Scalar theta2 = b[2];

    return Tangent(-theta1 * upsilon2[1] + theta2 * upsilon1[1],
                   theta1 * upsilon2[0] - theta2 * upsilon1[0], Scalar(0));
  }

  // Logarithmic map
  //
  // Computes the logarithm, the inverse of the group exponential which maps
  // element of the group (rigid body transformations) to elements of the
  // tangent space (twist).
  //
  // To be specific, this function computes ``vee(logmat(.))`` with
  // ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  // of SE(2).
  //
  SOPHUS_FUNC static Tangent log(SE2<Scalar> const& other) {
    Tangent upsilon_theta;
    SO2<Scalar> const& so2 = other.so2();
    Scalar theta = SO2<Scalar>::log(so2);
    upsilon_theta[2] = theta;
    Scalar halftheta = Scalar(0.5) * theta;
    Scalar halftheta_by_tan_of_halftheta;

    Vector2<Scalar> const& z = so2.unit_complex();
    Scalar real_minus_one = z.x() - Scalar(1.);
    if (std::abs(real_minus_one) < Constants<Scalar>::epsilon()) {
      halftheta_by_tan_of_halftheta =
          Scalar(1.) - Scalar(1. / 12) * theta * theta;
    } else {
      halftheta_by_tan_of_halftheta = -(halftheta * z.y()) / (real_minus_one);
    }
    Matrix<Scalar, 2, 2> V_inv;
    V_inv << halftheta_by_tan_of_halftheta, halftheta, -halftheta,
        halftheta_by_tan_of_halftheta;
    upsilon_theta.template head<2>() = V_inv * other.translation();
    return upsilon_theta;
  }

  // vee-operator
  //
  // It takes the 3x3-matrix representation ``Omega`` and maps it to the
  // corresponding 3-vector representation of Lie algebra.
  //
  // This is the inverse of the hat-operator, see above.
  //
  // Precondition: ``Omega`` must have the following structure:
  //
  //                |  0 -d  a |
  //                |  d  0  b |
  //                |  0  0  0 | .
  //
  SOPHUS_FUNC static Tangent vee(Transformation const& Omega) {
    SOPHUS_ENSURE(
        Omega.row(2).template lpNorm<1>() < Constants<Scalar>::epsilon(),
        "Omega: \n%", Omega);
    Tangent upsilon_omega;
    upsilon_omega.template head<2>() = Omega.col(2).template head<2>();
    upsilon_omega[2] = SO2<Scalar>::vee(Omega.template topLeftCorner<2, 2>());
    return upsilon_omega;
  }
};

// SE2 default type - Constructors and default storage for SE3 Type.
template <class Scalar_, int Options>
class SE2 : public SE2Base<SE2<Scalar_, Options>> {
  using Base = SE2Base<SE2<Scalar_, Options>>;

 public:
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using SO2Member = SO2<Scalar, Options>;
  using TranslationMember = Vector2<Scalar, Options>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor initialize rigid body motion to the identity.
  //
  SOPHUS_FUNC SE2() : translation_(Vector2<Scalar>::Zero()) {}

  // Copy constructor
  //
  template <class OtherDerived>
  SOPHUS_FUNC SE2(SE2Base<OtherDerived> const& other)
      : so2_(other.so2()), translation_(other.translation()) {
    static_assert(std::is_same<typename OtherDerived::Scalar, Scalar>::value,
                  "must be same Scalar type");
  }

  // Constructor from SO3 and translation vector
  //
  template <class OtherDerived, class D>
  SOPHUS_FUNC SE2(SO2Base<OtherDerived> const& so2,
                  Eigen::MatrixBase<D> const& translation)
      : so2_(so2), translation_(translation) {
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
  SE2(typename SO2<Scalar>::Transformation const& rotation_matrix,
      Point const& translation)
      : so2_(rotation_matrix), translation_(translation) {}

  // Constructor from rotation angle and translation vector.
  //
  SOPHUS_FUNC SE2(Scalar const& theta, Point const& translation)
      : so2_(theta), translation_(translation) {}

  // Constructor from complex number and translation vector
  //
  // Precondition: ``complex` must not be close to zero.
  SOPHUS_FUNC SE2(Vector2<Scalar> const& complex, Point const& translation)
      : so2_(complex), translation_(translation) {}

  // Constructor from 3x3 matrix
  //
  // Precondition: Rotation matrix needs to be orthogonal with determinant of 1.
  //               The last row must be (0, 0, 1).
  //
  SOPHUS_FUNC explicit SE2(Transformation const& T)
      : so2_(T.template topLeftCorner<2, 2>().eval()),
        translation_(T.template block<2, 1>(0, 2)) {}

  // Returns closest SE3 given arbirary 4x4 matrix.
  //
  SOPHUS_FUNC static SE2 fitToSE2(Matrix3<Scalar> const& T) {
    return SE2(SO2<Scalar>::fitToSO2(T.template block<2, 2>(0, 0)),
               T.template block<2, 1>(0, 2));
  }

  // Construct a translation only SE3 instance.
  //
  template <class T0, class T1>
  static SOPHUS_FUNC SE2 trans(T0 const& x, T1 const& y) {
    return SE2(SO2<Scalar>(), Vector2<Scalar>(x, y));
  }

  // Contruct x-axis translation.
  //
  static SOPHUS_FUNC SE2 transX(Scalar const& x) {
    return SE2::trans(x, Scalar(0));
  }

  // Contruct y-axis translation.
  //
  static SOPHUS_FUNC SE2 transY(Scalar const& y) {
    return SE2::trans(Scalar(0), y);
  }

  // Contruct pure rotation.
  //
  static SOPHUS_FUNC SE2 rot(Scalar const& x) {
    return SE2(SO2<Scalar>(x), Sophus::Vector2<Scalar>::Zero());
  }

  // This provides unsafe read/write access to internal data. SO(2) is
  // represented by a complex number (two parameters). When using direct write
  // access, the user needs to take care of that the complex number stays
  // normalized.
  //
  SOPHUS_FUNC Scalar* data() {
    // so2_ and translation_ are layed out sequentially with no padding
    return so2_.data();
  }

  // Const version of data() above.
  //
  SOPHUS_FUNC Scalar const* data() const {
    // so2_ and translation_ are layed out sequentially with no padding
    return so2_.data();
  }

  // Draw uniform sample from SE(3) manifold.
  //
  // Translations are drawn component-wise from the range [-1, 1].
  //
  template <class UniformRandomBitGenerator>
  static SE2 sampleUniform(UniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    return SE2(SO2<Scalar>::sampleUniform(generator),
               Vector2<Scalar>(uniform(generator), uniform(generator)));
  }

  // Accessor of SO3
  //
  SOPHUS_FUNC SO2Member& so2() { return so2_; }

  // Mutator of SO3
  //
  SOPHUS_FUNC SO2Member const& so2() const { return so2_; }

  // Mutator of translation vector
  //
  SOPHUS_FUNC TranslationMember& translation() { return translation_; }

  // Accessor of translation vector
  //
  SOPHUS_FUNC TranslationMember const& translation() const {
    return translation_;
  }

 protected:
  SO2Member so2_;
  TranslationMember translation_;
};

}  // end namespace

namespace Eigen {

// Specialization of Eigen::Map for ``SE2``.
//
// Allows us to wrap SE2 objects around POD array.
template <class Scalar_, int Options>
class Map<Sophus::SE2<Scalar_>, Options>
    : public Sophus::SE2Base<Map<Sophus::SE2<Scalar_>, Options>> {
  using Base = Sophus::SE2Base<Map<Sophus::SE2<Scalar_>, Options>>;

 public:
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  Map(Scalar* coeffs)
      : so2_(coeffs),
        translation_(coeffs + Sophus::SO2<Scalar>::num_parameters) {}

  // Mutator of SO3
  //
  SOPHUS_FUNC Map<Sophus::SO2<Scalar>, Options>& so2() { return so2_; }

  // Accessor of SO3
  //
  SOPHUS_FUNC Map<Sophus::SO2<Scalar>, Options> const& so2() const {
    return so2_;
  }

  // Mutator of translation vector
  //
  SOPHUS_FUNC Map<Sophus::Vector2<Scalar>, Options>& translation() {
    return translation_;
  }

  // Accessor of translation vector
  //
  SOPHUS_FUNC Map<Sophus::Vector2<Scalar>, Options> const& translation() const {
    return translation_;
  }

 protected:
  Map<Sophus::SO2<Scalar>, Options> so2_;
  Map<Sophus::Vector2<Scalar>, Options> translation_;
};

// Specialization of Eigen::Map for ``SE2 const``.
//
// Allows us to wrap SE2 objects around POD array.
template <class Scalar_, int Options>
class Map<Sophus::SE2<Scalar_> const, Options>
    : public Sophus::SE2Base<Map<Sophus::SE2<Scalar_> const, Options>> {
  using Base = Sophus::SE2Base<Map<Sophus::SE2<Scalar_> const, Options>>;

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
      : so2_(coeffs),
        translation_(coeffs + Sophus::SO2<Scalar>::num_parameters) {}

  // Accessor of SO3
  //
  SOPHUS_FUNC Map<Sophus::SO2<Scalar> const, Options> const& so2() const {
    return so2_;
  }

  // Accessor of translation vector
  //
  SOPHUS_FUNC Map<Sophus::Vector2<Scalar> const, Options> const& translation()
      const {
    return translation_;
  }

 protected:
  Map<Sophus::SO2<Scalar> const, Options> const so2_;
  Map<Sophus::Vector2<Scalar> const, Options> const translation_;
};
}

#endif
