#ifndef SOPHUS_RXSO2_HPP
#define SOPHUS_RXSO2_HPP

#include "so2.hpp"

namespace Sophus {
template <class Scalar_, int Options = 0>
class RxSO2;
using RxSO2d = RxSO2<double>;
using RxSO2f = RxSO2<float>;
}  // namespace Sophus

namespace Eigen {
namespace internal {

template <class Scalar_, int Options>
struct traits<Sophus::RxSO2<Scalar_, Options>> {
  using Scalar = Scalar_;
  using ComplexType = Sophus::Vector2<Scalar, Options>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::RxSO2<Scalar_>, Options>>
    : traits<Sophus::RxSO2<Scalar_, Options>> {
  using Scalar = Scalar_;
  using ComplexType = Map<Sophus::Vector2<Scalar>, Options>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::RxSO2<Scalar_> const, Options>>
    : traits<Sophus::RxSO2<Scalar_, Options> const> {
  using Scalar = Scalar_;
  using ComplexType = Map<Sophus::Vector2<Scalar> const, Options>;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {

// RxSO2 base type - implements RxSO2 class but is storage agnostic
//
// This class implements the group ``R+ x SO(2)``, the direct product of the
// group of positive scalar 2x2 matrices (= isomorph to the positive
// real numbers) and the two-dimensional special orthogonal group SO(2).
// Geometrically, it is the group of rotation and scaling in two dimensions.
// As a matrix groups, R+ x SO(2) consists of matrices of the form ``s * R``
// where ``R`` is an orthogonal matrix with ``det(R) = 1`` and ``s > 0``
// being a positive real number. In particular, it has the following form:
//
//  | s * cos(theta)  s * -sin(theta) |
//  | s * sin(theta)  s *  cos(theta) |
//
// where ``theta`` being the rotation angle. Internally, it is represented by
// the first column of the rotation matrix, or in other words by a non-zero
// complex number.
//
// R+ x SO(2) is not compact, but a commutative group. First it is not compact
// since the scale factor is not bound. Second it is commutative since
// ``sR(alpha, s1) * sR(beta, s2) = sR(beta, s2) * sR(alpha, s1)``,  simply
// because ``alpha + beta = beta + alpha`` and ``s1 * s2 = s2 * s1`` with
// ``alpha`` and ``beta`` being rotation angles and ``s1``, ``s2`` being scale
// factors.
//
// This class has the explicit class invariant that the scale ``s`` is not
// too close to zero. Strictly speaking, it must hold that:
//
//   ``complex().norm() >= Constants<Scalar>::epsilon()``.
//
// In order to obey this condition, group multiplication is implemented with
// saturation such that a product always has a scale which is equal or greater
// this threshold.
template <class Derived>
class RxSO2Base {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using ComplexType = typename Eigen::internal::traits<Derived>::ComplexType;

  // Degrees of freedom of manifold, number of dimensions in tangent space
  // (one for rotation and one for scaling).
  static int constexpr DoF = 2;
  // Number of internal parameters used (complex number is a tuple).
  static int constexpr num_parameters = 2;
  // Group transformations are 2x2 matrices.
  static int constexpr N = 2;
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
  // For RxSO(2), it simply returns the identity matrix.
  //
  SOPHUS_FUNC Adjoint Adj() const { return Adjoint::Identity(); }

  // Returns rotation angle.
  //
  SOPHUS_FUNC Scalar angle() const { return SO2<Scalar>(complex()).log(); }

  // Returns copy of instance casted to NewScalarType.
  //
  template <class NewScalarType>
  SOPHUS_FUNC RxSO2<NewScalarType> cast() const {
    return RxSO2<NewScalarType>(complex().template cast<NewScalarType>());
  }

  // This provides unsafe read/write access to internal data. RxSO(2) is
  // represented by a complex number (two parameters). When using direct
  // write access, the user needs to take care of that the complex number is not
  // set close to zero.
  //
  // Note: The first parameter represents the real part, while the
  // second parameter represent the imaginary part.
  //
  SOPHUS_FUNC Scalar* data() { return complex_nonconst().data(); }

  // Const version of data() above.
  //
  SOPHUS_FUNC Scalar const* data() const { return complex().data(); }

  // Returns group inverse.
  //
  SOPHUS_FUNC RxSO2<Scalar> inverse() const {
    Scalar squared_scale = complex().squaredNorm();
    return RxSO2<Scalar>(complex().x() / squared_scale,
                         -complex().y() / squared_scale);
  }

  // Logarithmic map
  //
  // Returns tangent space representation of the instance.
  //
  SOPHUS_FUNC Tangent log() const { return RxSO2<Scalar>::log(*this); }

  // Returns 2x2 matrix representation of the instance.
  //
  // For RxSO2, the matrix representation is an scaled orthogonal matrix ``sR``
  // with ``det(R)=s^2``, thus a scaled rotation matrix ``R``  with scale ``s``.
  //
  SOPHUS_FUNC Transformation matrix() const {
    Transformation sR;
    // clang-format off
    sR << complex()[0], -complex()[1],
          complex()[1],  complex()[0];
    // clang-format on
    return sR;
  }

  // Assignment operator.
  //
  template <class OtherDerived>
  SOPHUS_FUNC RxSO2Base<Derived>& operator=(
      RxSO2Base<OtherDerived> const& other) {
    complex() = other.complex();
    return *this;
  }

  // Group multiplication, which is rotation concatenation and scale
  // multiplication.
  //
  // Note: This function performs saturation for products close to zero in order
  // to ensure the class invariant.
  //
  SOPHUS_FUNC RxSO2<Scalar> operator*(RxSO2<Scalar> const& other) const {
    RxSO2<Scalar> result(*this);
    result *= other;
    return result;
  }

  // Group action on 2-points.
  //
  // This function rotates a 2 dimensional point ``p`` by the SO2 element
  // ``bar_R_foo`` (= rotation matrix) and scales it by the scale factor ``s``:
  //
  //   ``p_bar = s * (bar_R_foo * p_foo)``.
  //
  SOPHUS_FUNC Point operator*(Point const& p) const { return matrix() * p; }

  // Group action on lines.
  //
  // This function rotates a parametrized line ``l(t) = o + t * d`` by the SO2
  // element and scales it by the scale factor
  //
  // Origin ``o`` is rotated and scaled
  // Direction ``d`` is rotated (preserving it's norm)
  //
  SOPHUS_FUNC Line operator*(Line const& l) const {
    return Line((*this) * l.origin(), (*this) * l.direction() / scale());
  }

  // In-place group multiplication.
  //
  // Note: This function performs saturation for products close to zero in order
  // to ensure the class invariant.
  //
  SOPHUS_FUNC RxSO2Base<Derived>& operator*=(RxSO2<Scalar> const& other) {
    Scalar lhs_real = complex().x();
    Scalar lhs_imag = complex().y();
    Scalar const& rhs_real = other.complex().x();
    Scalar const& rhs_imag = other.complex().y();
    // complex multiplication
    complex_nonconst().x() = lhs_real * rhs_real - lhs_imag * rhs_imag;
    complex_nonconst().y() = lhs_real * rhs_imag + lhs_imag * rhs_real;

    Scalar squared_scale = complex_nonconst().squaredNorm();

    if (squared_scale <
        Constants<Scalar>::epsilon() * Constants<Scalar>::epsilon()) {
      // Saturation to ensure class invariant.
      complex_nonconst().normalize();
      complex_nonconst() *= Constants<Scalar>::epsilon();
    }
    return *this;
  }

  // Sets non-zero complex
  //
  // Precondition: ``z`` must not be close to zero.
  SOPHUS_FUNC void setComplex(Vector2<Scalar> const& z) {
    SOPHUS_ENSURE(z.squaredNorm() > Constants<Scalar>::epsilon() *
                                        Constants<Scalar>::epsilon(),
                  "Scale factor must be greater-equal epsilon.");
    static_cast<Derived*>(this)->complex_nonconst() = z;
  }

  // Accessor of complex.
  //
  SOPHUS_FUNC ComplexType const& complex() const {
    return static_cast<Derived const*>(this)->complex();
  }

  // Returns rotation matrix.
  //
  SOPHUS_FUNC Transformation rotationMatrix() const {
    ComplexType norm_quad = complex();
    norm_quad.normalize();
    return SO2<Scalar>(norm_quad).matrix();
  }

  // Returns scale.
  //
  SOPHUS_FUNC
  Scalar scale() const { return complex().norm(); }

  // Setter of rotation angle, leaves scale as is.
  //
  SOPHUS_FUNC void setAngle(Scalar const& theta) { setSO2(SO2<Scalar>(theta)); }

  // Setter of complex using rotation matrix ``R``, leaves scale as is.
  //
  // Precondition: ``R`` must be orthogonal with determinant of one.
  //
  SOPHUS_FUNC void setRotationMatrix(Transformation const& R) {
    setSO2(SO2<Scalar>(R));
  }

  // Sets scale and leaves rotation as is.
  //
  SOPHUS_FUNC void setScale(Scalar const& scale) {
    using std::sqrt;
    complex_nonconst().normalize();
    complex_nonconst() *= scale;
  }

  // Setter of complex number using scaled rotation matrix ``sR``.
  //
  // Precondition: The 2x2 matrix must be "scaled orthogonal"
  //               and have a positive determinant.
  //
  SOPHUS_FUNC void setScaledRotationMatrix(Transformation const& sR) {
    SOPHUS_ENSURE(isScaledOrthogonalAndPositive(sR),
                  "sR must be scaled orthogonal:\n %", sR);
    complex_nonconst() = sR.col(0);
  }

  // Setter of SO(2) rotations, leaves scale as is.
  //
  SOPHUS_FUNC void setSO2(SO2<Scalar> const& so2) {
    using std::sqrt;
    Scalar saved_scale = scale();
    complex_nonconst() = so2.unit_complex();
    complex_nonconst() *= saved_scale;
  }

  SOPHUS_FUNC SO2<Scalar> so2() const { return SO2<Scalar>(complex()); }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  // Derivative of Lie bracket with respect to first element.
  //
  // This function returns ``D_a [a, b]`` with ``D_a`` being the
  // differential operator with respect to ``a``, ``[a, b]`` being the lie
  // bracket of the Lie algebra rxso2.
  // See ``lieBracket()`` below.
  //
  SOPHUS_FUNC static Adjoint d_lieBracketab_by_d_a(Tangent const&) {
    Adjoint res;
    res.setZero();
    return res;
  }

  // Group exponential
  //
  // This functions takes in an element of tangent space (= rotation angle
  // plus logarithm of scale) and returns the corresponding element of the group
  // RxSO2.
  //
  // To be more specific, thixs function computes ``expmat(hat(theta))``
  // with ``expmat(.)`` being the matrix exponential and ``hat(.)`` being the
  // hat()-operator of RSO2.
  //
  SOPHUS_FUNC static RxSO2<Scalar> exp(Tangent const& a) {
    using std::exp;

    Scalar const theta = a[0];
    Scalar const sigma = a[1];
    Scalar s = exp(sigma);
    Vector2<Scalar> z = SO2<Scalar>::exp(theta).unit_complex();
    z *= s;
    return RxSO2<Scalar>(z);
  }

  // Returns the ith infinitesimal generators of ``R+ x SO(2)``.
  //
  // The infinitesimal generators of RxSO2 are:
  //
  //         |  0 -1 |
  //   G_0 = |  1  0 |
  //
  //         |  1  0 |
  //   G_1 = |  0  1 |
  //
  // Precondition: ``i`` must be 0, or 1.
  //
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 && i <= 1, "i should be 0 or 1.");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  // hat-operator
  //
  // It takes in the 2-vector representation ``a`` (= rotation angle plus
  // logarithm of scale) and  returns the corresponding matrix representation of
  // Lie algebra element.
  //
  // Formally, the ``hat()`` operator of RxSO2 is defined as
  //
  //   ``hat(.): R^2 -> R^{2x2},  hat(a) = sum_i a_i * G_i``  (for i=0,1,2)
  //
  // with ``G_i`` being the ith infinitesial generator of RxSO2.
  //
  // The corresponding inverse is the ``vee``-operator, see below.
  //
  SOPHUS_FUNC static Transformation hat(Tangent const& a) {
    Transformation A;
    // clang-format off
    A <<
       a(1), -a(0),
       a(0),  a(1);
    // clang-format on
    return A;
  }

  // Lie bracket
  //
  // It computes the Lie bracket of RxSO(2). To be more specific, it computes
  //
  //   ``[omega_1, omega_2]_rxso2 := vee([hat(omega_1), hat(omega_2)])``
  //
  // with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.) the
  // hat-operator and ``vee(.)`` the vee-operator of RxSO2.
  //
  SOPHUS_FUNC static Tangent lieBracket(Tangent const&, Tangent const&) {
    Vector2<Scalar> res;
    res.setZero();
    return res;
  }

  // Logarithmic map
  //
  // Computes the logarithm, the inverse of the group exponential which maps
  // element of the group (scaled rotation matrices) to elements of the tangent
  // space (rotation-vector plus logarithm of scale factor).
  //
  // To be specific, this function computes ``vee(logmat(.))`` with
  // ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  // of RxSO2.
  //
  SOPHUS_FUNC static Tangent log(RxSO2<Scalar> const& other) {
    using std::log;
    Scalar scale = other.scale();
    Tangent theta_sigma;
    theta_sigma[1] = log(scale);
    theta_sigma[0] = SO2<Scalar>::log(SO2<Scalar>(other.complex()));
    return theta_sigma;
  }

  // vee-operator
  //
  // It takes the 2x2-matrix representation ``Omega`` and maps it to the
  // corresponding vector representation of Lie algebra.
  //
  // This is the inverse of the hat-operator, see above.
  //
  // Precondition: ``Omega`` must have the following structure:
  //
  //                |  d -x |
  //                |  x  d | .
  //
  SOPHUS_FUNC static Tangent vee(Transformation const& Omega) {
    using std::abs;
    return Tangent(Omega(1, 0), Omega(0, 0));
  }

 protected:
  // Mutator of complex is private to ensure class invariant.
  //
  SOPHUS_FUNC ComplexType& complex_nonconst() {
    return static_cast<Derived*>(this)->complex_nonconst();
  }
};

// RxSO2 default type - Constructors and default storage for RxSO2 Type.
template <class Scalar_, int Options>
class RxSO2 : public RxSO2Base<RxSO2<Scalar_, Options>> {
  using Base = RxSO2Base<RxSO2<Scalar_, Options>>;

 public:
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using ComplexMember = Eigen::Matrix<Scalar, 2, 1, Options>;

  // ``Base`` is friend so complex_nonconst can be accessed from ``Base``.
  friend class RxSO2Base<RxSO2<Scalar_, Options>>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor initialize complex number to identity rotation and
  // scale.
  //
  SOPHUS_FUNC RxSO2() : complex_(Scalar(1), Scalar(0)) {}

  // Copy constructor
  //
  template <class OtherDerived>
  SOPHUS_FUNC RxSO2(RxSO2Base<OtherDerived> const& other)
      : complex_(other.complex()) {}

  // Constructor from scaled rotation matrix
  //
  // Precondition: rotation matrix need to be scaled orthogonal with determinant
  // of s^2.
  //
  SOPHUS_FUNC explicit RxSO2(Transformation const& sR) {
    this->setScaledRotationMatrix(sR);
  }

  // Constructor from scale factor and rotation matrix ``R``.
  //
  // Precondition: Rotation matrix ``R`` must to be orthogonal with determinant
  //               of 1 and ``scale`` must to be close to zero.
  //
  SOPHUS_FUNC RxSO2(Scalar const& scale, Transformation const& R)
      : RxSO2((scale * SO2<Scalar>(R).unit_complex()).eval()) {}

  // Constructor from scale factor and SO2
  //
  // Precondition: ``scale`` must to be close to zero.
  //
  SOPHUS_FUNC RxSO2(Scalar const& scale, SO2<Scalar> const& so2)
      : RxSO2((scale * so2.unit_complex()).eval()) {}

  // Constructor from complex number.
  //
  // Precondition: complex number must not be close to zero.
  //
  SOPHUS_FUNC explicit RxSO2(Vector2<Scalar> const& z) : complex_(z) {
    SOPHUS_ENSURE(complex_.squaredNorm() > Constants<Scalar>::epsilon() *
                                               Constants<Scalar>::epsilon(),
                  "Scale factor must be greater-equal epsilon: % vs %",
                  complex_.squaredNorm(),
                  Constants<Scalar>::epsilon() * Constants<Scalar>::epsilon());
  }

  // Constructor from complex number.
  //
  // Precondition: complex number must not be close to zero.
  //
  SOPHUS_FUNC explicit RxSO2(Scalar const& real, Scalar const& imag)
      : RxSO2(Vector2<Scalar>(real, imag)) {}

  // Draw uniform sample from RxSO(2) manifold.
  //
  // The scale factor is drawn uniformly in log2-space from [-1, 1],
  // hence the scale is in [0.5, 2)].
  //
  template <class UniformRandomBitGenerator>
  static RxSO2 sampleUniform(UniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    using std::exp2;
    return RxSO2(exp2(uniform(generator)),
                 SO2<Scalar>::sampleUniform(generator));
  }

  // Accessor of complex.
  //
  SOPHUS_FUNC ComplexMember const& complex() const { return complex_; }

 protected:
  SOPHUS_FUNC ComplexMember& complex_nonconst() { return complex_; }

  ComplexMember complex_;
};

}  // namespace Sophus

namespace Eigen {

// Specialization of Eigen::Map for ``RxSO2``.
//
// Allows us to wrap RxSO2 objects around POD array (e.g. external z style
// complex).
template <class Scalar_, int Options>
class Map<Sophus::RxSO2<Scalar_>, Options>
    : public Sophus::RxSO2Base<Map<Sophus::RxSO2<Scalar_>, Options>> {
  using Base = Sophus::RxSO2Base<Map<Sophus::RxSO2<Scalar_>, Options>>;

 public:
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  // ``Base`` is friend so complex_nonconst can be accessed from ``Base``.
  friend class Sophus::RxSO2Base<Map<Sophus::RxSO2<Scalar_>, Options>>;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar* coeffs) : complex_(coeffs) {}

  // Accessor of complex.
  //
  SOPHUS_FUNC
  Map<Sophus::Vector2<Scalar>, Options> const& complex() const {
    return complex_;
  }

 protected:
  SOPHUS_FUNC Map<Sophus::Vector2<Scalar>, Options>& complex_nonconst() {
    return complex_;
  }

  Map<Sophus::Vector2<Scalar>, Options> complex_;
};

// Specialization of Eigen::Map for ``RxSO2 const``.
//
// Allows us to wrap RxSO2 objects around POD array (e.g. external z style
// complex).
template <class Scalar_, int Options>
class Map<Sophus::RxSO2<Scalar_> const, Options>
    : public Sophus::RxSO2Base<Map<Sophus::RxSO2<Scalar_> const, Options>> {
  using Base = Sophus::RxSO2Base<Map<Sophus::RxSO2<Scalar_> const, Options>>;

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
  Map(Scalar const* coeffs) : complex_(coeffs) {}

  // Accessor of complex.
  //
  SOPHUS_FUNC
  Map<Sophus::Vector2<Scalar> const, Options> const& complex() const {
    return complex_;
  }

 protected:
  Map<Sophus::Vector2<Scalar> const, Options> const complex_;
};
}

#endif  // SOPHUS_RXSO2_HPP
