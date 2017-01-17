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

#ifndef SOPHUS_SIM3_HPP
#define SOPHUS_SIM3_HPP

#include "rxso3.hpp"

namespace Sophus {
template <typename _Scalar, int _Options = 0>
class Sim3Group;
typedef Sim3Group<double> Sim3d;
typedef Sim3Group<float> Sim3f;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<double, 7, 7> Matrix7d;
typedef Eigen::Matrix<float, 7, 1> Vector7f;
typedef Eigen::Matrix<float, 7, 7> Matrix7f;
}

namespace Eigen {
namespace internal {

template <typename _Scalar, int _Options>
struct traits<Sophus::Sim3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Eigen::Matrix<Scalar, 3, 1> TranslationType;
  typedef Sophus::RxSO3Group<Scalar> RxSO3Type;
};

template <typename _Scalar, int _Options>
struct traits<Map<Sophus::Sim3Group<_Scalar>, _Options>>
    : traits<Sophus::Sim3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<Eigen::Matrix<Scalar, 3, 1>, _Options> TranslationType;
  typedef Map<Sophus::RxSO3Group<Scalar>, _Options> RxSO3Type;
};

template <typename _Scalar, int _Options>
struct traits<Map<const Sophus::Sim3Group<_Scalar>, _Options>>
    : traits<const Sophus::Sim3Group<_Scalar, _Options>> {
  typedef _Scalar Scalar;
  typedef Map<const Eigen::Matrix<Scalar, 3, 1>, _Options> TranslationType;
  typedef Map<const Sophus::RxSO3Group<Scalar>, _Options> RxSO3Type;
};
}
}

namespace Sophus {

// Sim3 base type - implements Sim3 class but is storage agnostic.
//
// Sim(3) is the group of rotations  and translation and scaling in 3d. It is
// the semi-direct product of R+xSO(3) and the 3d Euclidean vector space.  The
// class is represented using a composition of RxSO3Group  for scaling plus
// rotation and a 3-vector for translation.
//
// Sim(3) is neither compact, nor a commutative group.
//
// See RxSO3Group for more details of the scaling + rotation representation in
// 3d.
//
template <typename Derived>
class Sim3GroupBase {
 public:
  /** \brief scalar type */
  typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;
  /** \brief translation reference type */
  typedef typename Eigen::internal::traits<Derived>::TranslationType&
      TranslationReference;
  /** \brief translation const reference type */
  typedef const typename Eigen::internal::traits<Derived>::TranslationType&
      ConstTranslationReference;
  /** \brief RxSO3 reference type */
  typedef typename Eigen::internal::traits<Derived>::RxSO3Type& RxSO3Reference;
  /** \brief RxSO3 const reference type */
  typedef const typename Eigen::internal::traits<Derived>::RxSO3Type&
      ConstRxSO3Reference;

  // Degrees of freedom of manifold, number of dimensions in tangent space
  // (three for translation, three for rotation and one for scaling).
  static const int DoF = 7;
  // Number of internal parameters used (4-tuple for quaternion, three for
  // translation).
  static const int num_parameters = 7;
  // Group transformations are 4x4 matrices.
  static const int N = 4;
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
  SOPHUS_FUNC Adjoint Adj() const {
    const Eigen::Matrix<Scalar, 3, 3>& R = rxso3().rotationMatrix();
    Adjoint res;
    res.setZero();
    res.block(0, 0, 3, 3) = scale() * R;
    res.block(0, 3, 3, 3) = SO3Group<Scalar>::hat(translation()) * R;
    res.block(0, 6, 3, 1) = -translation();
    res.block(3, 3, 3, 3) = R;
    res(6, 6) = 1;
    return res;
  }

  // Returns copy of instance casted to NewScalarType.
  //
  template <typename NewScalarType>
  SOPHUS_FUNC Sim3Group<NewScalarType> cast() const {
    return Sim3Group<NewScalarType>(
        rxso3().template cast<NewScalarType>(),
        translation().template cast<NewScalarType>());
  }

  // Returns group inverse.
  //
  SOPHUS_FUNC Sim3Group<Scalar> inverse() const {
    RxSO3Group<Scalar> invR = rxso3().inverse();
    return Sim3Group<Scalar>(invR,
                             invR * (translation() * static_cast<Scalar>(-1)));
  }

  // Logarithmic map
  //
  // Returns tangent space representation of the instance.
  //
  SOPHUS_FUNC Tangent log() const { return log(*this); }

  // Returns 4x4 matrix representation of the instance.
  //
  // It has the following form:
  //
  //   | s*R t |
  //   |  o  1 |
  //
  // where ``R`` is a 3x3 rotation matrix, ``s`` a scale factor, ``t`` a
  // translation 3-vector and ``o`` a 3-column vector of zeros.
  //
  SOPHUS_FUNC Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.setIdentity();
    homogenious_matrix.block(0, 0, 3, 3) = rxso3().matrix();
    homogenious_matrix.col(3).head(3) = translation();
    return homogenious_matrix;
  }

  // Returns the significant first three rows of the matrix above.
  //
  SOPHUS_FUNC Eigen::Matrix<Scalar, 3, 4> matrix3x4() const {
    Eigen::Matrix<Scalar, 3, 4> matrix;
    matrix.block(0, 0, 3, 3) = rxso3().matrix();
    matrix.col(3) = translation();
    return matrix;
  }

  // Assignment operator.
  //
  template <typename OtherDerived>
  SOPHUS_FUNC Sim3GroupBase<Derived>& operator=(
      const Sim3GroupBase<OtherDerived>& other) {
    rxso3() = other.rxso3();
    translation() = other.translation();
    return *this;
  }

  // Group multiplication, which is rotation plus scaling concatenation.
  //
  // Note: That scaling is calculated with saturation. See RxSO3Group for
  // details.
  //
  SOPHUS_FUNC Sim3Group<Scalar> operator*(
      const Sim3Group<Scalar>& other) const {
    Sim3Group<Scalar> result(*this);
    result *= other;
    return result;
  }

  // Group action on 3-points.
  //
  // This function rotates, scales and translates a three dimensional point
  // ``p`` by the Sim(3) element ``(bar_sR_foo, t_bar)`` (= similarity
  // transformation):
  //
  //   ``p_bar = bar_sR_foo * p_foo + t_bar``.
  //
  SOPHUS_FUNC Point operator*(const Point& p) const {
    return rxso3() * p + translation();
  }

  // In-place group multiplication.
  //
  SOPHUS_FUNC Sim3GroupBase<Derived>& operator*=(
      const Sim3Group<Scalar>& other) {
    translation() += (rxso3() * other.translation());
    rxso3() *= other.rxso3();
    return *this;
  }

  // Setter of non-zero quaternion.
  //
  // Precondition: ``quat`` must not be close to zero.
  //
  SOPHUS_FUNC void setQuaternion(const Eigen::Quaternion<Scalar>& quat) {
    rxso3().setQuaternion(quat);
  }

  // Accessor of quaternion.
  //
  SOPHUS_FUNC const Eigen::Quaternion<Scalar>& quaternion() const {
    return rxso3().quaternion();
  }

  // Returns Rotation matrix
  //
  SOPHUS_FUNC Eigen::Matrix<Scalar, 3, 3> rotationMatrix() const {
    return rxso3().rotationMatrix();
  }

  // Mutator of SO3 group.
  //
  SOPHUS_FUNC
  RxSO3Reference rxso3() { return static_cast<Derived*>(this)->rxso3(); }

  // Accessor of SO3 group.
  //
  SOPHUS_FUNC
  ConstRxSO3Reference rxso3() const {
    return static_cast<const Derived*>(this)->rxso3();
  }

  // Returns scale.
  //
  SOPHUS_FUNC Scalar scale() const { return rxso3().scale(); }

  // Setter of quaternion using rotation matrix ``R``, leaves scale as is.
  //
  SOPHUS_FUNC void setRotationMatrix(const Eigen::Matrix<Scalar, 3, 3>& R) {
    rxso3().setRotationMatrix(R);
  }

  // Sets scale and leaves rotation as is.
  //
  // Note: This function as a significant computational cost, since it has to
  // call the square root twice.
  //
  SOPHUS_FUNC
  void setScale(const Scalar& scale) { rxso3().setScale(scale); }

  // Setter of quaternion using scaled rotation matrix ``sR``.
  //
  // Precondition: The 3x3 matrix must be "scaled orthogonal"
  //               and have a positive determinant.
  //
  SOPHUS_FUNC void setScaledRotationMatrix(
      const Eigen::Matrix<Scalar, 3, 3>& sR) {
    rxso3().setScaledRotationMatrix(sR);
  }

  // Mutator of translation vector
  //
  SOPHUS_FUNC
  TranslationReference translation() {
    return static_cast<Derived*>(this)->translation();
  }

  // Accessor of translation vector
  //
  SOPHUS_FUNC
  ConstTranslationReference translation() const {
    return static_cast<const Derived*>(this)->translation();
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  // Derivative of Lie bracket with respect to first element.
  //
  // This function returns ``D_a [a, b]`` with ``D_a`` being the
  // differential operator with respect to ``a``, ``[a, b]`` being the lie
  // bracket of the Lie algebra sim(3).
  // See ``lieBracket()`` below.
  //
  SOPHUS_FUNC static Adjoint d_lieBracketab_by_d_a(const Tangent& b) {
    const Eigen::Matrix<Scalar, 3, 1>& upsilon2 = b.template head<3>();
    const Eigen::Matrix<Scalar, 3, 1>& omega2 = b.template segment<3>(3);
    Scalar sigma2 = b[6];

    Adjoint res;
    res.setZero();
    res.template topLeftCorner<3, 3>() =
        -SO3Group<Scalar>::hat(omega2) -
        sigma2 * Eigen::Matrix<Scalar, 3, 3>::Identity();
    res.template block<3, 3>(0, 3) = -SO3Group<Scalar>::hat(upsilon2);
    res.template topRightCorner<3, 1>() = upsilon2;
    res.template block<3, 3>(3, 3) = -SO3Group<Scalar>::hat(omega2);
    return res;
  }

  // Group exponential
  //
  // This functions takes in an element of tangent space and returns the
  // corresponding element of the group Sim(3).
  //
  // The first three components of ``a`` represent the translational part
  // ``upsilon`` in the tangent space of Sim(3), the following three components
  // of ``a`` represents the rotation vector ``omega`` and the final component
  // represents the logarithm of the scaling factor ``sigma``.
  // To be more specific, this function computes ``expmat(hat(a))`` with
  // ``expmat(.)`` being the matrix exponential and ``hat(.)`` the hat-operator
  // of Sim(3), see below.
  //
  SOPHUS_FUNC static Sim3Group<Scalar> exp(const Tangent& a) {
    const Eigen::Matrix<Scalar, 3, 1>& upsilon = a.segment(0, 3);
    const Eigen::Matrix<Scalar, 3, 1>& omega = a.segment(3, 3);
    Scalar sigma = a[6];
    Scalar theta;
    RxSO3Group<Scalar> rxso3 =
        RxSO3Group<Scalar>::expAndTheta(a.template tail<4>(), &theta);
    Eigen::Matrix<Scalar, 3, 3> Omega = SO3Group<Scalar>::hat(omega);
    Eigen::Matrix<Scalar, 3, 3> W = calcW(theta, sigma, rxso3.scale(), Omega);
    return Sim3Group<Scalar>(rxso3, W * upsilon);
  }

  // Returns the ith infinitesimal generators of Sim(3).
  //
  // The infinitesimal generators of Sim(3) are:
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
  //         |  1  0  0  0 |
  //   G_6 = |  0  1  0  0 |
  //         |  0  0  1  0 |
  //         |  0  0  0  0 |
  //
  // Precondition: ``i`` must be in [0, 6].
  //
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 || i <= 6, "i should be in range [0,6].");
    Tangent e;
    e.setZero();
    e[i] = static_cast<Scalar>(1);
    return hat(e);
  }

  // hat-operator
  //
  // It takes in the 7-vector representation and returns the corresponding
  // matrix representation of Lie algebra element.
  //
  // Formally, the ``hat()`` operator of Sim(3) is defined as
  //
  //   ``hat(.): R^7 -> R^{4x4},  hat(a) = sum_i a_i * G_i``  (for i=0,...,6)
  //
  // with ``G_i`` being the ith infinitesimal generator of Sim(3).
  //
  SOPHUS_FUNC static Transformation hat(const Tangent& a) {
    Transformation Omega;
    Omega.template topLeftCorner<3, 3>() =
        RxSO3Group<Scalar>::hat(a.template tail<4>());
    Omega.col(3).template head<3>() = a.template head<3>();
    Omega.row(3).setZero();
    return Omega;
  }

  // Lie bracket
  //
  // It computes the Lie bracket of Sim(3). To be more specific, it computes
  //
  //   ``[omega_1, omega_2]_sim3 := vee([hat(omega_1), hat(omega_2)])``
  //
  // with ``[A,B] := AB-BA`` being the matrix commutator, ``hat(.) the
  // hat-operator and ``vee(.)`` the vee-operator of Sim(3).
  //
  SOPHUS_FUNC static Tangent lieBracket(const Tangent& a, const Tangent& b) {
    const Eigen::Matrix<Scalar, 3, 1>& upsilon1 = a.template head<3>();
    const Eigen::Matrix<Scalar, 3, 1>& upsilon2 = b.template head<3>();
    const Eigen::Matrix<Scalar, 3, 1>& omega1 = a.template segment<3>(3);
    const Eigen::Matrix<Scalar, 3, 1>& omega2 = b.template segment<3>(3);
    Scalar sigma1 = a[6];
    Scalar sigma2 = b[6];

    Tangent res;
    res.template head<3>() = SO3Group<Scalar>::hat(omega1) * upsilon2 +
                             SO3Group<Scalar>::hat(upsilon1) * omega2 +
                             sigma1 * upsilon2 - sigma2 * upsilon1;
    res.template segment<3>(3) = omega1.cross(omega2);
    res[6] = static_cast<Scalar>(0);

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
  // of Sim(3).
  //
  SOPHUS_FUNC static Tangent log(const Sim3Group<Scalar>& other) {
    Tangent res;
    Scalar theta;
    Eigen::Matrix<Scalar, 4, 1> omega_sigma =
        RxSO3Group<Scalar>::logAndTheta(other.rxso3(), &theta);
    const Eigen::Matrix<Scalar, 3, 1>& omega = omega_sigma.template head<3>();
    Scalar sigma = omega_sigma[3];
    Eigen::Matrix<Scalar, 3, 3> W_inv =
        calcWInv(theta, sigma, other.scale(), SO3Group<Scalar>::hat(omega));
    res.segment(0, 3) = W_inv * other.translation();
    res.segment(3, 3) = omega;
    res[6] = sigma;
    return res;
  }

  // vee-operator
  //
  // It takes the 4x4-matrix representation ``Omega`` and maps it to the
  // corresponding 7-vector representation of Lie algebra.
  //
  // This is the inverse of the hat-operator, see above.
  //
  SOPHUS_FUNC static Tangent vee(const Transformation& Omega) {
    Tangent upsilon_omega_sigma;
    upsilon_omega_sigma.template head<3>() = Omega.col(3).template head<3>();
    upsilon_omega_sigma.template tail<4>() =
        RxSO3Group<Scalar>::vee(Omega.template topLeftCorner<3, 3>());
    return upsilon_omega_sigma;
  }

 private:
  static Eigen::Matrix<Scalar, 3, 3> calcW(
      const Scalar& theta, const Scalar& sigma, const Scalar& scale,
      const Eigen::Matrix<Scalar, 3, 3>& Omega) {
    static const Eigen::Matrix<Scalar, 3, 3> I =
        Eigen::Matrix<Scalar, 3, 3>::Identity();
    static const Scalar one = static_cast<Scalar>(1.);
    static const Scalar half = static_cast<Scalar>(1. / 2.);
    Eigen::Matrix<Scalar, 3, 3> Omega2 = Omega * Omega;

    Scalar A, B, C;
    if (abs(sigma) < Constants<Scalar>::epsilon()) {
      C = one;
      if (abs(theta) < Constants<Scalar>::epsilon()) {
        A = half;
        B = static_cast<Scalar>(1. / 6.);
      } else {
        Scalar theta_sq = theta * theta;
        A = (one - cos(theta)) / theta_sq;
        B = (theta - sin(theta)) / (theta_sq * theta);
      }
    } else {
      C = (scale - one) / sigma;
      if (abs(theta) < Constants<Scalar>::epsilon()) {
        Scalar sigma_sq = sigma * sigma;
        A = ((sigma - one) * scale + one) / sigma_sq;
        B = ((half * sigma * sigma - sigma + one) * scale) / (sigma_sq * sigma);
      } else {
        Scalar theta_sq = theta * theta;
        Scalar a = scale * sin(theta);
        Scalar b = scale * cos(theta);
        Scalar c = theta_sq + sigma * sigma;
        A = (a * sigma + (one - b) * theta) / (theta * c);
        B = (C - ((b - one) * sigma + a * theta) / (c)) * one / (theta_sq);
      }
    }
    return A * Omega + B * Omega2 + C * I;
  }

  static Eigen::Matrix<Scalar, 3, 3> calcWInv(
      const Scalar& theta, const Scalar& sigma, const Scalar& scale,
      const Eigen::Matrix<Scalar, 3, 3>& Omega) {
    static const Eigen::Matrix<Scalar, 3, 3> I =
        Eigen::Matrix<Scalar, 3, 3>::Identity();
    static const Scalar half = static_cast<Scalar>(0.5);
    static const Scalar one = static_cast<Scalar>(1.);
    static const Scalar two = static_cast<Scalar>(2.);
    const Eigen::Matrix<Scalar, 3, 3> Omega2 = Omega * Omega;
    const Scalar scale_sq = scale * scale;
    const Scalar theta_sq = theta * theta;
    const Scalar sin_theta = sin(theta);
    const Scalar cos_theta = cos(theta);

    Scalar a, b, c;
    if (abs(sigma * sigma) < Constants<Scalar>::epsilon()) {
      c = one - half * sigma;
      a = -half;
      if (abs(theta_sq) < Constants<Scalar>::epsilon()) {
        b = Scalar(1. / 12.);
      } else {
        b = (theta * sin_theta + two * cos_theta - two) /
            (two * theta_sq * (cos_theta - one));
      }
    } else {
      const Scalar scale_cu = scale_sq * scale;
      c = sigma / (scale - one);
      if (abs(theta_sq) < Constants<Scalar>::epsilon()) {
        a = (-sigma * scale + scale - one) / ((scale - one) * (scale - one));
        b = (scale_sq * sigma - two * scale_sq + scale * sigma + two * scale) /
            (two * scale_cu - Scalar(6) * scale_sq + Scalar(6) * scale - two);
      } else {
        const Scalar s_sin_theta = scale * sin_theta;
        const Scalar s_cos_theta = scale * cos_theta;
        a = (theta * s_cos_theta - theta - sigma * s_sin_theta) /
            (theta * (scale_sq - two * s_cos_theta + one));
        b = -scale *
            (theta * s_sin_theta - theta * sin_theta + sigma * s_cos_theta -
             scale * sigma + sigma * cos_theta - sigma) /
            (theta_sq * (scale_cu - two * scale * s_cos_theta - scale_sq +
                         two * s_cos_theta + scale - one));
      }
    }
    return a * Omega + b * Omega2 + c * I;
  }
};

// Sim3 default type - Constructors and default storage for Sim3 Type.
template <typename _Scalar, int _Options>
class Sim3Group : public Sim3GroupBase<Sim3Group<_Scalar, _Options>> {
  typedef Sim3GroupBase<Sim3Group<_Scalar, _Options>> Base;

 public:
  typedef typename Eigen::internal::traits<Sim3Group<_Scalar, _Options>>::Scalar
      Scalar;
  typedef
      typename Eigen::internal::traits<Sim3Group<_Scalar, _Options>>::RxSO3Type&
          RxSO3Reference;
  typedef const typename Eigen::internal::traits<
      Sim3Group<_Scalar, _Options>>::RxSO3Type& ConstRxSO3Reference;
  typedef typename Eigen::internal::traits<
      Sim3Group<_Scalar, _Options>>::TranslationType& TranslationReference;
  typedef const typename Eigen::internal::traits<
      Sim3Group<_Scalar, _Options>>::TranslationType& ConstTranslationReference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Default constructor initialize similiraty transform to the identity.
  //
  SOPHUS_FUNC Sim3Group() : translation_(Eigen::Matrix<Scalar, 3, 1>::Zero()) {}

  // Copy constructor
  //
  template <typename OtherDerived>
  SOPHUS_FUNC Sim3Group(const Sim3GroupBase<OtherDerived>& other)
      : rxso3_(other.rxso3()), translation_(other.translation()) {}

  // Constructor from RxSO3 and translation vector
  //
  template <typename OtherDerived>
  SOPHUS_FUNC Sim3Group(const RxSO3GroupBase<OtherDerived>& rxso3,
                        const Point& translation)
      : rxso3_(rxso3), translation_(translation) {}

  // Constructor from quaternion and translation vector.
  //
  // Precondition: quaternion must not be close to zero.
  //
  SOPHUS_FUNC Sim3Group(const Eigen::Quaternion<Scalar>& quaternion,
                        const Point& translation)
      : rxso3_(quaternion), translation_(translation) {}

  // Constructor from 4x4 matrix
  //
  // Precondition: Top-left 3x3 matrix needs to be "scaled-orthogonal" with
  //               positive determinant. The last row must be (0, 0, 0, 1).
  //
  SOPHUS_FUNC explicit Sim3Group(const Eigen::Matrix<Scalar, 4, 4>& T)
      : rxso3_(T.template topLeftCorner<3, 3>()),
        translation_(T.template block<3, 1>(0, 3)) {}

  // This provides unsafe read/write access to internal data. Sim(3) is
  // represented by an Eigen::Quaternion (four parameters) and a 3-vector. When
  // using direct write access, the user needs to take care of that the
  // quaternion is not set close to zero.
  //
  SOPHUS_FUNC Scalar* data() {
    // rxso3_ and translation_ are laid out sequentially with no padding
    return rxso3_.data();
  }

  // Const version of data() above.
  //
  SOPHUS_FUNC
  const Scalar* data() const {
    // rxso3_ and translation_ are laid out sequentially with no padding
    return rxso3_.data();
  }

  // Accessor of RxSO3
  //
  SOPHUS_FUNC
  RxSO3Reference rxso3() { return rxso3_; }

  // Mutator of RxSO3
  //
  SOPHUS_FUNC
  ConstRxSO3Reference rxso3() const { return rxso3_; }

  // Mutator of translation vector
  //
  SOPHUS_FUNC
  TranslationReference translation() { return translation_; }

  // Accessor of translation vector
  //
  SOPHUS_FUNC
  ConstTranslationReference translation() const { return translation_; }

 protected:
  Sophus::RxSO3Group<Scalar> rxso3_;
  Eigen::Matrix<Scalar, 3, 1> translation_;
};

}  // namespace Sophus

namespace Eigen {

// Specialization of Eigen::Map for Sim3Group.
//
// Allows us to wrap Sim3 objects around POD array.
template <typename _Scalar, int _Options>
class Map<Sophus::Sim3Group<_Scalar>, _Options>
    : public Sophus::Sim3GroupBase<Map<Sophus::Sim3Group<_Scalar>, _Options>> {
  typedef Sophus::Sim3GroupBase<Map<Sophus::Sim3Group<_Scalar>, _Options>> Base;

 public:
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  typedef typename Eigen::internal::traits<Map>::TranslationType&
      TranslationReference;
  typedef const typename Eigen::internal::traits<Map>::TranslationType&
      ConstTranslationReference;
  typedef typename Eigen::internal::traits<Map>::RxSO3Type& RxSO3Reference;
  typedef const typename Eigen::internal::traits<Map>::RxSO3Type&
      ConstRxSO3Reference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  Map(Scalar* coeffs)
      : rxso3_(coeffs),
        translation_(coeffs + Sophus::RxSO3Group<Scalar>::num_parameters) {}

  // Mutator of RxSO3
  //
  SOPHUS_FUNC RxSO3Reference rxso3() { return rxso3_; }

  // Accessor of RxSO3
  //
  SOPHUS_FUNC ConstRxSO3Reference rxso3() const { return rxso3_; }

  // Mutator of translation vector
  //
  SOPHUS_FUNC TranslationReference translation() { return translation_; }

  // Accessor of translation vector
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return translation_;
  }

 protected:
  Map<Sophus::RxSO3Group<Scalar>, _Options> rxso3_;
  Map<Eigen::Matrix<Scalar, 3, 1>, _Options> translation_;
};

// Specialization of Eigen::Map for ``const Sim3GroupBase``
//
// Allows us to wrap RxSO3 objects around POD array.
template <typename _Scalar, int _Options>
class Map<const Sophus::Sim3Group<_Scalar>, _Options>
    : public Sophus::Sim3GroupBase<
          Map<const Sophus::Sim3Group<_Scalar>, _Options>> {
  typedef Sophus::Sim3GroupBase<Map<const Sophus::Sim3Group<_Scalar>, _Options>>
      Base;

 public:
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  typedef const typename Eigen::internal::traits<Map>::TranslationType&
      ConstTranslationReference;
  typedef const typename Eigen::internal::traits<Map>::RxSO3Type&
      ConstRxSO3Reference;

  typedef typename Base::Transformation Transformation;
  typedef typename Base::Point Point;
  typedef typename Base::Tangent Tangent;
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(const Scalar* coeffs)
      : rxso3_(coeffs),
        translation_(coeffs + Sophus::RxSO3Group<Scalar>::num_parameters) {}

  SOPHUS_FUNC Map(const Scalar* trans_coeffs, const Scalar* rot_coeffs)
      : rxso3_(rot_coeffs), translation_(trans_coeffs) {}

  // Accessor of RxSO3
  //
  SOPHUS_FUNC ConstRxSO3Reference rxso3() const { return rxso3_; }

  // Accessor of translation vector
  //
  SOPHUS_FUNC ConstTranslationReference translation() const {
    return translation_;
  }

 protected:
  const Map<const Sophus::RxSO3Group<Scalar>, _Options> rxso3_;
  const Map<const Eigen::Matrix<Scalar, 3, 1>, _Options> translation_;
};
}

#endif
