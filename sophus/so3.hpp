#ifndef SOPHUS_SO3_HPP
#define SOPHUS_SO3_HPP

#include "types.hpp"

// Include only the selective set of Eigen headers that we need.
// This helps when using Sophus with unusual compilers, like nvcc.
#include <Eigen/src/Geometry/OrthoMethods.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/src/Geometry/RotationBase.h>

namespace Sophus {
template <class Scalar_, int Options = 0>
class SO3;
using SO3d = SO3<double>;
using SO3f = SO3<float>;
}  // namespace Sophus

namespace Eigen {
namespace internal {

template <class Scalar_, int Options>
struct traits<Sophus::SO3<Scalar_, Options>> {
  using Scalar = Scalar_;
  using QuaternionType = Eigen::Quaternion<Scalar>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::SO3<Scalar_>, Options>>
    : traits<Sophus::SO3<Scalar_, Options>> {
  using Scalar = Scalar_;
  using QuaternionType = Map<Eigen::Quaternion<Scalar>, Options>;
};

template <class Scalar_, int Options>
struct traits<Map<Sophus::SO3<Scalar_> const, Options>>
    : traits<Sophus::SO3<Scalar_, Options> const> {
  using Scalar = Scalar_;
  using QuaternionType = Map<Eigen::Quaternion<Scalar> const, Options>;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {

// SO3 base type - implements SO3 class but is storage agnostic.
//
// SO(3) is the group of rotations in 3d. As a matrix group, it is the set of
// matrices which are orthogonal such that ``R * R' = I`` (with ``R'`` being the
// transpose of ``R``) and have a positive determinant. In particular, the
// determinant is 1. Internally, the group is represented as a unit quaternion.
// Unit quaternion can be seen as members of the special unitary group SU(2).
// SU(2) is a double cover of SO(3). Hence, for every rotation matrix ``R``,
// there exists two unit quaternion: ``(r, v)`` and ``(r, -v)``, with ``r`` the
// real part and ``v`` being the imaginary 3-vector part of the quaternion.
//
// SO(3) is a compact, but non-commutative group. First it is compact since the
// set of rotation matrices is a closed and bounded set. Second it is
// non-commutative since the equation ``R_1 * R_2 = R_2 * R_1`` does not hold in
// general. For example rotating an object by some degrees about its ``x``-axis
// and then by some degrees about its y axis, does not lead to the same
// orienation when rotation first about ``y`` and then about ``x``.
//
// Class invairant: The 2-norm of ``unit_quaternion`` must be close to 1.
// Technically speaking, it must hold that:
//
//   ``|unit_quaternion().squaredNorm() - 1| <= Constants<Scalar>::epsilon()``.
template <class Derived>
class SO3Base {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using QuaternionType =
      typename Eigen::internal::traits<Derived>::QuaternionType;

  // Degrees of freedom of group, number of dimensions in tangent space.
  static int constexpr DoF = 3;
  // Number of internal parameters used (quaternion is a 4-tuple).
  static int constexpr num_parameters = 4;
  // Group transformations are 3x3 matrices.
  static int constexpr N = 3;
  using Transformation = Matrix<Scalar, N, N>;
  using Point = Vector3<Scalar>;
  using Tangent = Vector<Scalar, DoF>;
  using Adjoint = Matrix<Scalar, DoF, DoF>;

  // Adjoint transformation
  //
  // This function return the adjoint transformation ``Ad`` of the group
  // element ``A`` such that for all ``x`` it holds that
  // ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  //
  // For SO(3), it simply returns the rotation matrix corresponding to ``A``.
  //
  SOPHUS_FUNC Adjoint Adj() const { return matrix(); }

  // Returns copy of instance casted to NewScalarType.
  //
  template <class NewScalarType>
  SOPHUS_FUNC SO3<NewScalarType> cast() const {
    return SO3<NewScalarType>(unit_quaternion().template cast<NewScalarType>());
  }

  // This provides unsafe read/write access to internal data. SO(3) is
  // represented by an Eigen::Quaternion (four parameters). When using direct
  // write access, the user needs to take care of that the quaternion stays
  // normalized.
  //
  // Note: The first three Scalars represent the imaginary parts, while the
  // forth Scalar represent the real part.
  //
  SOPHUS_FUNC Scalar* data() {
    return unit_quaternion_nonconst().coeffs().data();
  }

  // Const version of data() above.
  //
  SOPHUS_FUNC Scalar const* data() const {
    return unit_quaternion().coeffs().data();
  }

  // Returns ``*this`` times the ith generator of internal SU(2) representation.
  //
  SOPHUS_FUNC Vector<Scalar, num_parameters> internalMultiplyByGenerator(
      int i) const {
    Vector<Scalar, num_parameters> res;
    Eigen::Quaternion<Scalar> internal_gen_q;
    internalGenerator(i, &internal_gen_q);
    res.template head<4>() = (unit_quaternion() * internal_gen_q).coeffs();
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
  SOPHUS_FUNC SO3<Scalar> inverse() const {
    return SO3<Scalar>(unit_quaternion().conjugate());
  }

  // Logarithmic map
  //
  // Returns tangent space representation (= rotation vector) of the instance.
  //
  SOPHUS_FUNC Tangent log() const { return SO3<Scalar>::log(*this); }

  // It re-normalizes ``unit_quaternion`` to unit length.
  //
  // Note: Because of the class invariant, there is typically no need to call
  // this function directly.
  //
  SOPHUS_FUNC void normalize() {
    Scalar length = unit_quaternion_nonconst().norm();
    SOPHUS_ENSURE(length >= Constants<Scalar>::epsilon(),
                  "Quaternion (%) should not be close to zero!",
                  unit_quaternion_nonconst().coeffs().transpose());
    unit_quaternion_nonconst().coeffs() /= length;
  }

  // Returns 3x3 matrix representation of the instance.
  //
  // For SO(3), the matrix representation is an orthogonal matrix ``R`` with
  // ``det(R)=1``, thus the so-called "rotation matrix".
  //
  SOPHUS_FUNC Transformation matrix() const {
    return unit_quaternion().toRotationMatrix();
  }

  // Assignment operator.
  //
  template <class OtherDerived>
  SOPHUS_FUNC SO3Base<Derived>& operator=(SO3Base<OtherDerived> const& other) {
    unit_quaternion_nonconst() = other.unit_quaternion();
    return *this;
  }

  // Group multiplication, which is rotation concatenation.
  //
  SOPHUS_FUNC SO3<Scalar> operator*(SO3<Scalar> const& other) const {
    SO3<Scalar> result(*this);
    result *= other;
    return result;
  }

  // Group action on 3-points.
  //
  // This function rotates a 3 dimensional point ``p`` by the SO3 element
  //  ``bar_R_foo`` (= rotation matrix): ``p_bar = bar_R_foo * p_foo``.
  //
  // Since SO3 is internally represented by a unit quaternion ``q``, it is
  // implemented as ``p_bar = q * p_foo * q^{*}``
  // with ``q^{*}`` being the quaternion conjugate of ``q``.
  //
  // Geometrically, ``p``  is rotated by angle ``|omega|`` around the
  // axis ``omega/|omega|`` with ``omega := vee(log(bar_R_foo))``.
  //
  // For ``vee``-operator, see below.
  //
  SOPHUS_FUNC Point operator*(Point const& p) const {
    return unit_quaternion()._transformVector(p);
  }

  // In-place group multiplication.
  //
  SOPHUS_FUNC SO3Base<Derived>& operator*=(SO3<Scalar> const& other) {
    unit_quaternion_nonconst() *= other.unit_quaternion();

    Scalar squared_norm = unit_quaternion().squaredNorm();

    // We can assume that the squared-norm is close to 1 since we deal with a
    // unit quaternion. Due to numerical precision issues, there might
    // be a small drift after pose concatenation. Hence, we need to renormalizes
    // the quaternion here.
    // Since squared-norm is close to 1, we do not need to calculate the costly
    // square-root, but can use an approximation around 1 (see
    // http://stackoverflow.com/a/12934750 for details).
    if (squared_norm != Scalar(1.0)) {
      unit_quaternion_nonconst().coeffs() *=
          Scalar(2.0) / (Scalar(1.0) + squared_norm);
    }
    return *this;
  }

  // Takes in quaternion, and normalizes it.
  //
  // Precondition: The quaternion must not be close to zero.
  //
  SOPHUS_FUNC void setQuaternion(Eigen::Quaternion<Scalar> const& quaternion) {
    unit_quaternion_nonconst() = quaternion;
    normalize();
  }

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC QuaternionType const& unit_quaternion() const {
    return static_cast<Derived const*>(this)->unit_quaternion();
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  // Derivative of Lie bracket with respect to first element.
  //
  // This function returns ``D_a [a, b]`` with ``D_a`` being the
  // differential operator with respect to ``a``, ``[a, b]`` being the lie
  // bracket of the Lie algebra so3.
  // See ``lieBracket()`` below.
  //
  SOPHUS_FUNC static Adjoint d_lieBracketab_by_d_a(Tangent const& b) {
    return -hat(b);
  }

  // Group exponential
  //
  // This functions takes in an element of tangent space (= rotation vector
  // ``omega``) and returns the corresponding element of the group SO(3).
  //
  // To be more specific, this function computes ``expmat(hat(omega))``
  // with ``expmat(.)`` being the matrix exponential and ``hat(.)`` being the
  // hat()-operator of SO(3).
  //
  SOPHUS_FUNC static SO3<Scalar> exp(Tangent const& omega) {
    Scalar theta;
    return expAndTheta(omega, &theta);
  }

  // As above, but also returns ``theta = |omega|`` as out-parameter.
  //
  // Precondition: ``theta`` must not be ``nullptr``.
  //
  SOPHUS_FUNC static SO3<Scalar> expAndTheta(Tangent const& omega,
                                             Scalar* theta) {
    using std::sqrt;
    using std::abs;
    using std::sin;
    using std::cos;
    Scalar theta_sq = omega.squaredNorm();
    *theta = sqrt(theta_sq);
    Scalar half_theta = Scalar(0.5) * (*theta);

    Scalar imag_factor;
    Scalar real_factor;
    if ((*theta) < Constants<Scalar>::epsilon()) {
      Scalar theta_po4 = theta_sq * theta_sq;
      imag_factor = Scalar(0.5) - Scalar(1.0 / 48.0) * theta_sq +
                    Scalar(1.0 / 3840.0) * theta_po4;
      real_factor =
          Scalar(1) - Scalar(0.5) * theta_sq + Scalar(1.0 / 384.0) * theta_po4;
    } else {
      Scalar sin_half_theta = sin(half_theta);
      imag_factor = sin_half_theta / (*theta);
      real_factor = cos(half_theta);
    }

    SO3<Scalar> q;
    q.unit_quaternion_nonconst() = Eigen::Quaternion<Scalar>(
        real_factor, imag_factor * omega.x(), imag_factor * omega.y(),
        imag_factor * omega.z());
    SOPHUS_ENSURE(abs(q.unit_quaternion().squaredNorm() - Scalar(1)) <
                      Sophus::Constants<Scalar>::epsilon(),
                  "SO3::exp failed! omega: %, real: %, img: %",
                  omega.transpose(), real_factor, imag_factor);
    return q;
  }

  // Returns the ith infinitesimal generators of SO(3).
  //
  // The infinitesimal generators of SO(3) are:
  //
  //         |  0  0  0 |
  //   G_0 = |  0  0 -1 |
  //         |  0  1  0 |
  //
  //         |  0  0  1 |
  //   G_1 = |  0  0  0 |
  //         | -1  0  0 |
  //
  //         |  0 -1  0 |
  //   G_2 = |  1  0  0 |
  //         |  0  0  0 |
  //
  // Precondition: ``i`` must be 0, 1 or 2.
  //
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 && i <= 2, "i should be in range [0,2].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }

  // Returns the ith generator of internal SU(2) representation.
  //
  // Precondition: ``i`` must be 0, 1 or 2.
  //
  SOPHUS_FUNC static void internalGenerator(
      int i, Eigen::Quaternion<Scalar>* internal_gen_q) {
    SOPHUS_ENSURE(i >= 0 && i <= 2, "i should be in range [0,2]");
    SOPHUS_ENSURE(internal_gen_q != NULL,
                  "internal_gen_q must not be the null pointer");
    // Factor of 0.5 since SU(2) is a double cover of SO(3).
    internal_gen_q->coeffs()[i] = Scalar(0.5);
  }

  // hat-operator
  //
  // It takes in the 3-vector representation ``omega`` (= rotation vector) and
  // returns the corresponding matrix representation of Lie algebra element.
  //
  // Formally, the ``hat()`` operator of SO(3) is defined as
  //
  //   ``hat(.): R^3 -> R^{3x3},  hat(omega) = sum_i omega_i * G_i``
  //   (for i=0,1,2)
  //
  // with ``G_i`` being the ith infinitesimal generator of SO(3).
  //
  // The corresponding inverse is the ``vee``-operator, see below.
  //
  SOPHUS_FUNC static Transformation hat(Tangent const& omega) {
    Transformation Omega;
    // clang-format off
    Omega <<
        Scalar(0), -omega(2),  omega(1),
         omega(2), Scalar(0), -omega(0),
        -omega(1),  omega(0), Scalar(0);
    // clang-format on
    return Omega;
  }

  // Lie bracket
  //
  // It computes the Lie bracket of SO(3). To be more specific, it computes
  //
  //   ``[omega_1, omega_2]_so3 := vee([hat(omega_1), hat(omega_2)])``
  //
  // with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.) the
  // hat-operator and ``vee(.)`` the vee-operator of SO3.
  //
  // For the Lie algebra so3, the Lie bracket is simply the cross product:
  //
  // ``[omega_1, \omega_2]_so3 = omega_1 x \omega_2.``
  //
  SOPHUS_FUNC static Tangent lieBracket(Tangent const& omega1,
                                        Tangent const& omega2) {
    return omega1.cross(omega2);
  }

  // Logarithmic map
  //
  // Computes the logarithm, the inverse of the group exponential which maps
  // element of the group (rotation matrices) to elements of the tangent space
  // (rotation-vector).
  //
  // To be specific, this function computes ``vee(logmat(.))`` with
  // ``logmat(.)`` being the matrix logarithm and ``vee(.)`` the vee-operator
  // of SO(3).
  //
  SOPHUS_FUNC static Tangent log(SO3<Scalar> const& other) {
    Scalar theta;
    return logAndTheta(other, &theta);
  }

  // As above, but also returns ``theta = |omega|`` as out-parameter.
  //
  SOPHUS_FUNC static Tangent logAndTheta(SO3<Scalar> const& other,
                                         Scalar* theta) {
    using std::sqrt;
    using std::atan;
    using std::abs;
    Scalar squared_n = other.unit_quaternion().vec().squaredNorm();
    Scalar n = sqrt(squared_n);
    Scalar w = other.unit_quaternion().w();

    Scalar two_atan_nbyw_by_n;

    // Atan-based log thanks to
    //
    // C. Hertzberg et al.:
    // "Integrating Generic Sensor Fusion Algorithms with Sound State
    // Representation through Encapsulation of Manifolds"
    // Information Fusion, 2011

    if (n < Constants<Scalar>::epsilon()) {
      // If quaternion is normalized and n=0, then w should be 1;
      // w=0 should never happen here!
      SOPHUS_ENSURE(abs(w) >= Constants<Scalar>::epsilon(),
                    "Quaternion (%) should be normalized!",
                    other.unit_quaternion().coeffs().transpose());
      Scalar squared_w = w * w;
      two_atan_nbyw_by_n =
          Scalar(2) / w - Scalar(2) * (squared_n) / (w * squared_w);
    } else {
      if (abs(w) < Constants<Scalar>::epsilon()) {
        if (w > Scalar(0)) {
          two_atan_nbyw_by_n = Constants<Scalar>::pi() / n;
        } else {
          two_atan_nbyw_by_n = -Constants<Scalar>::pi() / n;
        }
      } else {
        two_atan_nbyw_by_n = Scalar(2) * atan(n / w) / n;
      }
    }

    *theta = two_atan_nbyw_by_n * n;

    return two_atan_nbyw_by_n * other.unit_quaternion().vec();
  }

  // vee-operator
  //
  // It takes the 3x3-matrix representation ``Omega`` and maps it to the
  // corresponding vector representation of Lie algebra.
  //
  // This is the inverse of the hat-operator, see above.
  //
  // Precondition: ``Omega`` must have the following structure:
  //
  //                |  0 -c  b |
  //                |  c  0 -a |
  //                | -b  a  0 | .
  //
  SOPHUS_FUNC static Tangent vee(Transformation const& Omega) {
    using std::abs;
    SOPHUS_ENSURE(
        Omega.diagonal().template lpNorm<1>() < Constants<Scalar>::epsilon(),
        "Omega: \n%", Omega);
    SOPHUS_ENSURE(abs(Omega(2, 1) + Omega(1, 2)) < Constants<Scalar>::epsilon(),
                  "Omega: %s", Omega);
    SOPHUS_ENSURE(abs(Omega(0, 2) + Omega(2, 0)) < Constants<Scalar>::epsilon(),
                  "Omega: %s", Omega);
    SOPHUS_ENSURE(abs(Omega(1, 0) + Omega(0, 1)) < Constants<Scalar>::epsilon(),
                  "Omega: %s", Omega);
    return Tangent(Omega(2, 1), Omega(0, 2), Omega(1, 0));
  }

 private:
  // Mutator of unit_quaternion is private to ensure class invariant. That is
  // the quaternion must stay close to unit length.
  //
  SOPHUS_FUNC QuaternionType& unit_quaternion_nonconst() {
    return static_cast<Derived*>(this)->unit_quaternion_nonconst();
  }
};

// SO3 default type - Constructors and default storage for SO3 Type.
template <class Scalar_, int Options>
class SO3 : public SO3Base<SO3<Scalar_, Options>> {
  using Base = SO3Base<SO3<Scalar_, Options>>;

 public:
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  // ``Base`` is friend so unit_quaternion_nonconst can be accessed from
  // ``Base``.
  friend class SO3Base<SO3<Scalar, Options>>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor initialize unit quaternion to identity rotation.
  //
  SOPHUS_FUNC SO3()
      : unit_quaternion_(Scalar(1), Scalar(0), Scalar(0), Scalar(0)) {}

  // Copy constructor
  //
  template <class OtherDerived>
  SOPHUS_FUNC SO3(SO3Base<OtherDerived> const& other)
      : unit_quaternion_(other.unit_quaternion()) {}

  // Constructor from rotation matrix
  //
  // Precondition: rotation matrix needs to be orthogonal with determinant of 1.
  //
  SOPHUS_FUNC SO3(Transformation const& R) : unit_quaternion_(R) {}

  // Constructor from quaternion
  //
  // Precondition: quaternion must not be close to zero.
  //
  SOPHUS_FUNC explicit SO3(Eigen::Quaternion<Scalar> const& quat)
      : unit_quaternion_(quat) {
    Base::normalize();
  }

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC Eigen::Quaternion<Scalar> const& unit_quaternion() const {
    return unit_quaternion_;
  }

 protected:
  // Mutator of unit_quaternion is protected to ensure class invariant.
  //
  SOPHUS_FUNC Eigen::Quaternion<Scalar>& unit_quaternion_nonconst() {
    return unit_quaternion_;
  }

  Eigen::Quaternion<Scalar> unit_quaternion_;
};

template <class Scalar, int Options = 0>
using SO3Group[[deprecated]] = SO3<Scalar, Options>;

}  // namespace Sophus

namespace Eigen {

// Specialization of Eigen::Map for ``SO3``.
//
// Allows us to wrap SO3 objects around POD array (e.g. external c style
// quaternion).
template <class Scalar_, int Options>
class Map<Sophus::SO3<Scalar_>, Options>
    : public Sophus::SO3Base<Map<Sophus::SO3<Scalar_>, Options>> {
  using Base = Sophus::SO3Base<Map<Sophus::SO3<Scalar_>, Options>>;

 public:
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  // ``Base`` is friend so unit_quaternion_nonconst can be accessed from
  // ``Base``.
  friend class Sophus::SO3Base<Map<Sophus::SO3<Scalar_>, Options>>;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar* coeffs) : unit_quaternion_(coeffs) {}

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC Map<Eigen::Quaternion<Scalar>, Options> const& unit_quaternion()
      const {
    return unit_quaternion_;
  }

 protected:
  // Mutator of unit_quaternion is protected to ensure class invariant.
  //
  SOPHUS_FUNC Map<Eigen::Quaternion<Scalar>, Options>&
  unit_quaternion_nonconst() {
    return unit_quaternion_;
  }

  Map<Eigen::Quaternion<Scalar>, Options> unit_quaternion_;
};

// Specialization of Eigen::Map for ``SO3 const``.
//
// Allows us to wrap SO3 objects around POD array (e.g. external c style
// quaternion).
template <class Scalar_, int Options>
class Map<Sophus::SO3<Scalar_> const, Options>
    : public Sophus::SO3Base<Map<Sophus::SO3<Scalar_> const, Options>> {
  using Base = Sophus::SO3Base<Map<Sophus::SO3<Scalar_> const, Options>>;

 public:
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar const* coeffs) : unit_quaternion_(coeffs) {}

  // Accessor of unit quaternion.
  //
  SOPHUS_FUNC Map<Eigen::Quaternion<Scalar> const, Options> const&
  unit_quaternion() const {
    return unit_quaternion_;
  }

 protected:
  // Mutator of unit_quaternion is protected to ensure class invariant.
  //
  Map<Eigen::Quaternion<Scalar> const, Options> const unit_quaternion_;
};
}

#endif
