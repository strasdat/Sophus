// This file is part of Sophus.
//
// Copyright 2012-2013 Hauke Strasdat
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef SOPHUS_RXSO3_HPP
#define SOPHUS_RXSO3_HPP

#include "so3.hpp"

namespace Sophus {
template <typename _Scalar, int _Options = 0>
class RxSO3Group;
typedef RxSO3Group<double> RxSO3d; /**< double precision RxSO3 */
typedef RxSO3Group<float> RxSO3f;  /**< single precision RxSO3 */
}  // namespace Sophus

namespace Eigen {
namespace internal {

template <typename _Scalar, int _Options>
struct traits<Sophus::RxSO3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Eigen::Quaternion<Scalar> QuaternionType;
};

template <typename _Scalar, int _Options>
struct traits<Map<Sophus::RxSO3Group<_Scalar>, _Options>>
    : traits<Sophus::RxSO3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<Eigen::Quaternion<Scalar>, _Options> QuaternionType;
};

template <typename _Scalar, int _Options>
struct traits<Map<const Sophus::RxSO3Group<_Scalar>, _Options>>
    : traits<const Sophus::RxSO3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<const Eigen::Quaternion<Scalar>, _Options> QuaternionType;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {

// RxSO3 base type - implements RxSO3 class but is storage agnostic
//
// This class implements the group ``R+ x SO(3)``, the direct product of the
// group of positive scalar 3x3 matrices (= isomorph to the positive
// real numbers) and the three-dimensional special orthogonal group SO(3).
// Geometrically, it is the group of rotation and scaling in three dimensions.
// As a matrix groups, RxSO3 consists of matrices of the form ``s * R``
// where ``R`` is an orthogonal matrix with ``det(R)=1`` and ``s > 0``
// being a positive real number.
//
// Internally, RxSO3 is represented by the group of non-zero quaternions.
// In particular, the scale equals the squared(!) norm of the quaternion ``q``,
// ``s = |q|^2``. This is a most compact representation since the degrees of
// freedom (DoF) of RxSO3 (=4) equals the number of internal parameters (=4).
//
// This class has the explicit class invariant that the scale ``s`` is not close
// too close to zero. Strictly speaking, it must hold that:
//
//   ``quaternion().squaredNorm() >= Constants<Scalar>::epsilon()``.
//
// In order to obey this condition, group multiplication is implemented with
// saturation such that a product always has a scale which is equal or greater
// this threshold.
template <typename Derived>
class RxSO3GroupBase {
 public:
  typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;
  typedef typename Eigen::internal::traits<Derived>::QuaternionType&
      QuaternionReference;
  typedef const typename Eigen::internal::traits<Derived>::QuaternionType&
      ConstQuaternionReference;

  // Degrees of freedom of manifold, number of dimensions in tangent space
  // (three for rotation and one for scaling).
  static const int DoF = 4;
  // Number of internal parameters used (quaternion is a 4-tuple).
  static const int num_parameters = 4;
  // Group transformations are 3x3 matrices.
  static const int N = 3;
  typedef Eigen::Matrix<Scalar, N, N> Transformation;
  typedef Eigen::Matrix<Scalar, 3, 1> Point;
  typedef Eigen::Matrix<Scalar, DoF, 1> Tangent;
  typedef Eigen::Matrix<Scalar, DoF, DoF> Adjoint;

  // Adjoint transformation
  //
  // This function return the adjoint transformation ``Ad`` of the group
  // element ``A`` such that for all ``x`` it holds that
  // ``hat(Ad_A * x) = A * hat(x) A^{-1}``. See hat-operator below.
  //
  // For RxSO(3), it simply returns the rotation matrix corresponding to ``A``.
  //
  SOPHUS_FUNC Adjoint Adj() const {
    Adjoint res;
    res.setIdentity();
    res.template topLeftCorner<3, 3>() = rotationMatrix();
    return res;
  }

  // Returns copy of instance casted to NewScalarType.
  //
  template <typename NewScalarType>
  SOPHUS_FUNC RxSO3Group<NewScalarType> cast() const {
    return RxSO3Group<NewScalarType>(
        quaternion().template cast<NewScalarType>());
  }

  // This provides unsafe read/write access to internal data. RxSO(3) is
  // represented by an Eigen::Quaternion (four parameters). When using direct
  // write access, the user needs to take care of that the quaternion is not
  // set close to zero.
  //
  // Note: The first three Scalars represent the imaginary parts, while the
  // forth Scalar represent the real part.
  //
  SOPHUS_FUNC Scalar* data() { return quaternion_nonconst().coeffs().data(); }

  // Const version of data() above.
  //
  SOPHUS_FUNC const Scalar* data() const {
    return quaternion().coeffs().data();
  }

  // Returns group inverse.
  //
  SOPHUS_FUNC RxSO3Group<Scalar> inverse() const {
    return RxSO3Group<Scalar>(quaternion().inverse());
  }

  // Logarithmic map
  //
  // Returns tangent space representation of the instance.
  //
  SOPHUS_FUNC Tangent log() const { return RxSO3Group<Scalar>::log(*this); }

  /**
   * \returns 3x3 matrix representation of instance
   *
   * For RxSO3, the matrix representation is a scaled orthogonal
   * matrix \f$ sR \f$ with \f$ det(sR)=s^3 \f$, thus a scaled rotation
   * matrix \f$ R \f$  with scale s.
   */

  // Returns 3x3 matrix representation of the instance.
  //
  // For RxSO3, the matrix representation is an scaled orthogonal matrix ``sR``
  // with ``det(R)=s^3``, thus a scaled rotation matrix ``R``  with scale ``s``.
  //
  SOPHUS_FUNC Transformation matrix() const {
    Transformation sR;

    const Scalar vx_sq = quaternion().vec().x() * quaternion().vec().x();
    const Scalar vy_sq = quaternion().vec().y() * quaternion().vec().y();
    const Scalar vz_sq = quaternion().vec().z() * quaternion().vec().z();
    const Scalar w_sq = quaternion().w() * quaternion().w();
    const Scalar two_vx = Scalar(2) * quaternion().vec().x();
    const Scalar two_vy = Scalar(2) * quaternion().vec().y();
    const Scalar two_vz = Scalar(2) * quaternion().vec().z();
    const Scalar two_vx_vy = two_vx * quaternion().vec().y();
    const Scalar two_vx_vz = two_vx * quaternion().vec().z();
    const Scalar two_vx_w = two_vx * quaternion().w();
    const Scalar two_vy_vz = two_vy * quaternion().vec().z();
    const Scalar two_vy_w = two_vy * quaternion().w();
    const Scalar two_vz_w = two_vz * quaternion().w();

    sR(0, 0) = vx_sq - vy_sq - vz_sq + w_sq;
    sR(1, 0) = two_vx_vy + two_vz_w;
    sR(2, 0) = two_vx_vz - two_vy_w;

    sR(0, 1) = two_vx_vy - two_vz_w;
    sR(1, 1) = -vx_sq + vy_sq - vz_sq + w_sq;
    sR(2, 1) = two_vx_w + two_vy_vz;

    sR(0, 2) = two_vx_vz + two_vy_w;
    sR(1, 2) = -two_vx_w + two_vy_vz;
    sR(2, 2) = -vx_sq - vy_sq + vz_sq + w_sq;
    return sR;
  }

  // Assignment operator.
  //
  template <typename OtherDerived>
  SOPHUS_FUNC RxSO3GroupBase<Derived>& operator=(
      const RxSO3GroupBase<OtherDerived>& other) {
    quaternion() = other.quaternion();
    return *this;
  }

  // Group multiplication, which is rotation concatenation and scale
  // multiplication.
  //
  // Note: This function performs saturation for products close to zero in order
  // to ensure the class invariant.
  //
  SOPHUS_FUNC RxSO3Group<Scalar> operator*(
      const RxSO3Group<Scalar>& other) const {
    RxSO3Group<Scalar> result(*this);
    result *= other;
    return result;
  }

  // Group action on 3-points.
  //
  // This function rotates a 3 dimensional point ``p`` by the SO3 element
  //  ``bar_R_foo`` (= rotation matrix) and scales it by the scale factor ``s``:
  //
  //   ``p_bar = s * (bar_R_foo * p_foo)``.
  //
  SOPHUS_FUNC Point operator*(const Point& p) const {
    // Follows http://eigen.tuxfamily.org/bz/show_bug.cgi?id=459
    Scalar scale = quaternion().squaredNorm();
    Point two_vec_cross_p = quaternion().vec().cross(p);
    two_vec_cross_p += two_vec_cross_p;
    return scale * p + (quaternion().w() * two_vec_cross_p +
                        quaternion().vec().cross(two_vec_cross_p));
  }

  // In-place group multiplication.
  //
  // Note: This function performs saturation for products close to zero in order
  // to ensure the class invariant.
  //
  SOPHUS_FUNC RxSO3GroupBase<Derived>& operator*=(
      const RxSO3Group<Scalar>& other) {
    using std::sqrt;

    quaternion_nonconst() *= other.quaternion();
    Scalar scale = this->scale();
    if (scale < Constants<Scalar>::epsilon()) {
      SOPHUS_ENSURE(scale > 0, "Scale must be greater zero.");
      // Saturation to ensure class invariant.
      quaternion_nonconst().normalize();
      quaternion_nonconst().coeffs() *= sqrt(Constants<Scalar>::epsilon());
    }
    return *this;
  }

  // Sets non-zero quaternion
  //
  // Precondition: ``quat`` must not be close to zero.
  SOPHUS_FUNC void setQuaternion(ConstQuaternionReference quat) {
    SOPHUS_ENSURE(quat.squaredNorm() > Constants<Scalar>::epsilon() *
                                           Constants<Scalar>::epsilon(),
                  "Scale factor must be greater-equal epsilon.");
    static_cast<Derived*>(this)->quaternion_nonconst() = quat;
  }

  // Accessor of quaternion.
  //
  SOPHUS_FUNC ConstQuaternionReference quaternion() const {
    return static_cast<const Derived*>(this)->quaternion();
  }

  // Returns rotation matrix.
  //
  SOPHUS_FUNC Transformation rotationMatrix() const {
    Eigen::Quaternion<Scalar> norm_quad = quaternion();
    norm_quad.normalize();
    return norm_quad.toRotationMatrix();
  }

  // Returns scale.
  //
  SOPHUS_FUNC
  Scalar scale() const { return quaternion().squaredNorm(); }

  // Setter of quaternion using rotation matrix ``R``, leaves scale as is.
  //
  SOPHUS_FUNC void setRotationMatrix(const Transformation& R) {
    Scalar saved_scale = scale();
    quaternion() = R;
    quaternion() *= saved_scale;
  }

  // Sets scale and leaves rotation as is.
  //
  // Note: This function as a significant computational cost, since it has to
  // call the square root twice.
  //
  SOPHUS_FUNC
  void setScale(const Scalar& scale) {
    using std::sqrt;
    quaternion_nonconst().normalize();
    quaternion_nonconst().coeffs() *= sqrt(scale);
  }

  // Setter of quaternion using scaled rotation matrix ``sR``.
  //
  // Precondition: The 3x3 matrix must be "scaled orthogonal"
  //               and have a positive determinant.
  //
  SOPHUS_FUNC void setScaledRotationMatrix(const Transformation& sR) {
    Transformation squared_sR = sR * sR.transpose();
    Scalar squared_scale =
        static_cast<Scalar>(1. / 3.) *
        (squared_sR(0, 0) + squared_sR(1, 1) + squared_sR(2, 2));
    SOPHUS_ENSURE(squared_scale > Constants<Scalar>::epsilon() *
                                      Constants<Scalar>::epsilon(),
                  "Scale factor must be greater-equal epsilon.");
    Scalar scale = sqrt(squared_scale);
    quaternion_nonconst() = sR / scale;
    quaternion_nonconst().coeffs() *= sqrt(scale);
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  // Derivative of Lie bracket with respect to first element.
  //
  // This function returns ``D_a [a, b]`` with ``D_a`` being the
  // differential operator with respect to ``a``, ``[a, b]`` being the lie
  // bracket of the Lie algebra rxso3.
  // See ``lieBracket()`` below.
  //
  SOPHUS_FUNC static Adjoint d_lieBracketab_by_d_a(const Tangent& b) {
    Adjoint res;
    res.setZero();
    res.template topLeftCorner<3, 3>() =
        -SO3Group<Scalar>::hat(b.template head<3>());
    return res;
  }

  // Group exponential
  //
  // This functions takes in an element of tangent space (= rotation 3-vector
  // plus logarithm of scale) and returns the corresponding element of the group
  // RxSO3.
  //
  // To be more specific, thixs function computes ``expmat(hat(omega))``
  // with ``expmat(.)`` being the matrix exponential and ``hat(.)`` being the
  // hat()-operator of RSO3.
  //
  SOPHUS_FUNC static RxSO3Group<Scalar> exp(const Tangent& a) {
    Scalar theta;
    return expAndTheta(a, &theta);
  }

  // As above, but also returns ``theta = |omega|`` as out-parameter.
  //
  // Precondition: ``theta`` must not be ``nullptr``.
  //
  SOPHUS_FUNC static RxSO3Group<Scalar> expAndTheta(const Tangent& a,
                                                    Scalar* theta) {
    SOPHUS_ENSURE(theta != nullptr, "must not be nullptr.");
    using std::exp;
    using std::sqrt;

    const Eigen::Matrix<Scalar, 3, 1>& omega = a.template head<3>();
    Scalar sigma = a[3];
    Scalar sqrt_scale = sqrt(exp(sigma));
    Eigen::Quaternion<Scalar> quat =
        SO3Group<Scalar>::expAndTheta(omega, theta).unit_quaternion();
    quat.coeffs() *= sqrt_scale;
    return RxSO3Group<Scalar>(quat);
  }

  // Returns the ith infinitesimal generators of ``R+ x SO(3)``.
  //
  // The infinitesimal generators of RxSO3 are:
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
  //         |  1  0  0 |
  //   G_2 = |  0  1  0 |
  //         |  0  0  1 |
  //
  // Precondition: ``i`` must be 0, 1, 2 or 3.
  //
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 && i <= 3, "i should be in range [0,3].");
    Tangent e;
    e.setZero();
    e[i] = static_cast<Scalar>(1);
    return hat(e);
  }

  // hat-operator
  //
  // It takes in the 4-vector representation ``a`` (= rotation vector plus
  // logarithm of scale) and  returns the corresponding matrix representation of
  // Lie algebra element.
  //
  // Formally, the ``hat()`` operator of RxSO3 is defined as
  //
  //   ``hat(.): R^4 -> R^{3x3},  hat(a) = sum_i a_i * G_i``  (for i=0,1,2,3)
  //
  // with ``G_i`` being the ith infinitesial generator of RxSO3.
  //
  // The corresponding inverse is the ``vee``-operator, see below.
  //
  SOPHUS_FUNC static Transformation hat(const Tangent& a) {
    Transformation A;
    // clang-format off
    A <<
       a(3), -a(2),  a(1),
       a(2),  a(3), -a(0),
      -a(1),  a(0),  a(3);
    // clang-format on
    return A;
  }

  // Lie bracket
  //
  // It computes the Lie bracket of RxSO(3). To be more specific, it computes
  //
  //   ``[omega_1, omega_2]_rxso3 := vee([hat(omega_1), hat(omega_2)])``
  //
  // with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.) the
  // hat-operator and ``vee(.)`` the vee-operator of RxSO3.
  //
  SOPHUS_FUNC static Tangent lieBracket(const Tangent& a, const Tangent& b) {
    const Eigen::Matrix<Scalar, 3, 1>& omega1 = a.template head<3>();
    const Eigen::Matrix<Scalar, 3, 1>& omega2 = b.template head<3>();
    Eigen::Matrix<Scalar, 4, 1> res;
    res.template head<3>() = omega1.cross(omega2);
    res[3] = static_cast<Scalar>(0);
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
  // of RxSO3.
  //
  SOPHUS_FUNC static Tangent log(const RxSO3Group<Scalar>& other) {
    Scalar theta;
    return logAndTheta(other, &theta);
  }

  // As above, but also returns ``theta = |omega|`` as out-parameter.
  //
  SOPHUS_FUNC static Tangent logAndTheta(const RxSO3Group<Scalar>& other,
                                         Scalar* theta) {
    using std::log;

    Scalar scale = other.quaternion().squaredNorm();
    Tangent omega_sigma;
    omega_sigma[3] = log(scale);
    omega_sigma.template head<3>() = SO3Group<Scalar>::logAndTheta(
        SO3Group<Scalar>(other.quaternion()), theta);
    return omega_sigma;
  }

  // vee-operator
  //
  // It takes the 3x3-matrix representation ``Omega`` and maps it to the
  // corresponding vector representation of Lie algebra.
  //
  // This is the inverse of the hat-operator, see above.
  //
  SOPHUS_FUNC static Tangent vee(const Transformation& Omega) {
    return Tangent(static_cast<Scalar>(0.5) * (Omega(2, 1) - Omega(1, 2)),
                   static_cast<Scalar>(0.5) * (Omega(0, 2) - Omega(2, 0)),
                   static_cast<Scalar>(0.5) * (Omega(1, 0) - Omega(0, 1)),
                   static_cast<Scalar>(1. / 3.) *
                       (Omega(0, 0) + Omega(1, 1) + Omega(2, 2)));
  }

 protected:
  // Mutator of quaternion is private to ensure class invariant.
  //
  SOPHUS_FUNC QuaternionReference quaternion_nonconst() {
    return static_cast<Derived*>(this)->quaternion_nonconst();
  }
};

// RxSO3 default type - Constructors and default storage for RxSO3 Type.
template <typename _Scalar, int _Options>
class RxSO3Group : public RxSO3GroupBase<RxSO3Group<_Scalar, _Options>> {
  typedef RxSO3GroupBase<RxSO3Group<_Scalar, _Options>> Base;

 public:
  typedef
      typename Eigen::internal::traits<RxSO3Group<_Scalar, _Options>>::Scalar
          Scalar;
  typedef typename Eigen::internal::traits<
      RxSO3Group<_Scalar, _Options>>::QuaternionType& QuaternionReference;
  typedef const typename Eigen::internal::traits<
      RxSO3Group<_Scalar, _Options>>::QuaternionType& ConstQuaternionReference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  // ``Base`` is friend so quaternion_nonconst can be accessed from ``Base``.
  friend class RxSO3GroupBase<RxSO3Group<_Scalar, _Options>>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor initialize quaternion to identity rotation and scale.
  //
  SOPHUS_FUNC RxSO3Group()
      : quaternion_(static_cast<Scalar>(1), static_cast<Scalar>(0),
                    static_cast<Scalar>(0), static_cast<Scalar>(0)) {}

  // Copy constructor
  //
  template <typename OtherDerived>
  SOPHUS_FUNC RxSO3Group(const RxSO3GroupBase<OtherDerived>& other)
      : quaternion_(other.quaternion()) {}

  // Constructor from scaled rotation matrix
  //
  // Precondition: rotation matrix need to be scaled orthogonal with determinant
  // of s^3.
  //
  SOPHUS_FUNC explicit RxSO3Group(const Transformation& sR) {
    this->setScaledRotationMatrix(sR);
  }

  // Constructor from scale factor and rotation matrix ``R``.
  //
  // Precondition: Rotation matrix ``R`` must to be orthogonal with determinant
  //               of 1 and ``scale`` must to be close to zero.
  //
  SOPHUS_FUNC RxSO3Group(const Scalar& scale, const Transformation& R)
      : quaternion_(R) {
    SOPHUS_ENSURE(scale >= Constants<Scalar>::epsilon(),
                  "Scale factor must be greater-equal epsilon.");
    quaternion_.normalize();
    quaternion_.coeffs() *= scale;
  }

  // Constructor from scale factor and SO3
  //
  // Precondition: ``scale`` must to be close to zero.
  //
  SOPHUS_FUNC RxSO3Group(const Scalar& scale, const SO3Group<Scalar>& so3)
      : quaternion_(so3.unit_quaternion()) {
    SOPHUS_ENSURE(scale >= Constants<Scalar>::epsilon(),
                  "Scale factor must be greater-equal epsilon.");
    quaternion_.normalize();
    quaternion_.coeffs() *= scale;
  }

  // Constructor from quaternion
  //
  // Precondition: quaternion must not be close to zero.
  //
  SOPHUS_FUNC explicit RxSO3Group(const Eigen::Quaternion<Scalar>& quat)
      : quaternion_(quat) {
    SOPHUS_ENSURE(quaternion_.squaredNorm() > Constants<Scalar>::epsilon(),
                  "Scale factor must be greater-equal epsilon.");
  }

  // Accessor of quaternion.
  //
  SOPHUS_FUNC ConstQuaternionReference quaternion() const {
    return quaternion_;
  }

 protected:
  SOPHUS_FUNC QuaternionReference quaternion_nonconst() { return quaternion_; }

  Eigen::Quaternion<Scalar> quaternion_;
};

}  // namespace Sophus

namespace Eigen {

// Specialization of Eigen::Map for RxSO3Group
//
// Allows us to wrap RxSO3 objects around POD array (e.g. external c style
// quaternion).
template <typename _Scalar, int _Options>
class Map<Sophus::RxSO3Group<_Scalar>, _Options>
    : public Sophus::RxSO3GroupBase<
          Map<Sophus::RxSO3Group<_Scalar>, _Options>> {
  typedef Sophus::RxSO3GroupBase<Map<Sophus::RxSO3Group<_Scalar>, _Options>>
      Base;

 public:
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  typedef typename Eigen::internal::traits<Map>::QuaternionType&
      QuaternionReference;
  typedef const typename Eigen::internal::traits<Map>::QuaternionType&
      ConstQuaternionReference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  // ``Base`` is friend so quaternion_nonconst can be accessed from ``Base``.
  friend class Sophus::SO3GroupBase<Map<Sophus::SO3Group<_Scalar>, _Options>>;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar* coeffs) : quaternion_(coeffs) {}

  // Accessor of quaternion.
  //
  SOPHUS_FUNC
  ConstQuaternionReference quaternion() const { return quaternion_; }

 protected:
  SOPHUS_FUNC QuaternionReference quaternion_nonconst() { return quaternion_; }

  Map<Eigen::Quaternion<Scalar>, _Options> quaternion_;
};

// Specialization of Eigen::Map for ``const RxSO3GroupBase``
//
// Allows us to wrap RxSO3 objects around POD array (e.g. external c style
// quaternion).
template <typename _Scalar, int _Options>
class Map<const Sophus::RxSO3Group<_Scalar>, _Options>
    : public Sophus::RxSO3GroupBase<
          Map<const Sophus::RxSO3Group<_Scalar>, _Options>> {
  typedef Sophus::RxSO3GroupBase<
      Map<const Sophus::RxSO3Group<_Scalar>, _Options>>
      Base;

 public:
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  typedef const typename Eigen::internal::traits<Map>::QuaternionType&
      ConstQuaternionReference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  Map(const Scalar* coeffs) : quaternion_(coeffs) {}

  // Accessor of quaternion.
  //
  SOPHUS_FUNC
  ConstQuaternionReference quaternion() const { return quaternion_; }

 protected:
  const Map<const Eigen::Quaternion<Scalar>, _Options> quaternion_;
};
}

#endif  // SOPHUS_RXSO3_HPP
