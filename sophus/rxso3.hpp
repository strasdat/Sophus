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
#include "sophus.hpp"

////////////////////////////////////////////////////////////////////////////
// Forward Declarations / typedefs
////////////////////////////////////////////////////////////////////////////

namespace Sophus {
template <typename _Scalar, int _Options = 0>
class RxSO3Group;
typedef RxSO3Group<double> RxSO3d; /**< double precision RxSO3 */
typedef RxSO3Group<float> RxSO3f;  /**< single precision RxSO3 */
}

////////////////////////////////////////////////////////////////////////////
// Eigen Traits (For querying derived types in CRTP hierarchy)
////////////////////////////////////////////////////////////////////////////

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
}
}

namespace Sophus {
/**
 * \brief RxSO3 base type - implements RxSO3 class but is storage agnostic
 *
 * This class implements the group \f$ R^{+} \times SO3 \f$ (RxSO3), the direct
 * product of the group of positive scalar matrices (=isomorph to the positive
 * real numbers) and the three-dimensional special orthogonal group SO3.
 * Geometrically, it is the group of rotation and scaling in three dimensions.
 * As a matrix groups, RxSO3 consists of matrices of the form \f$ s\cdot R \f$
 * where \f$ R \f$ is an orthognal matrix with \f$ det(R)=1 \f$ and \f$ s>0 \f$
 * be a positive real number.
 *
 * Internally, RxSO3 is represented by the group of non-zero quaternions.
 * In particular, the scale equals the squared(!) norm of the quaternion,
 * \f$ s = |q|^2 \f$. This is a most compact representation since the degrees of
 * freedom (DoF) of RxSO3 (=4) equals the number of internal parameters (=4).
 *
 * This class has the explicit class invariant that the scale \f$ s \f$ is
 * greater equal ``SophusConstant<Scalar>::epsilon()``. In order to obey
 * this condition, group multiplication is implemented with saturation
 * such that a product always has a scale which is equal or greater this
 * threshold.
 */
template <typename Derived>
class RxSO3GroupBase {
 public:
  /** \brief scalar type, use with care since this might be a Map type  */
  typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;
  /** \brief quaternion reference type */
  typedef typename Eigen::internal::traits<Derived>::QuaternionType&
      QuaternionReference;
  /** \brief quaternion const reference type */
  typedef const typename Eigen::internal::traits<Derived>::QuaternionType&
      ConstQuaternionReference;

  /** \brief degree of freedom of group
   *         (three for rotation and one for scaling) */
  static const int DoF = 4;
  /** \brief number of internal parameters used
   *         (quaternion for rotation and scaling) */
  static const int num_parameters = 4;
  /** \brief group transformations are NxN matrices */
  static const int N = 3;
  /** \brief group transfomation type */
  typedef Eigen::Matrix<Scalar, N, N> Transformation;
  /** \brief point type */
  typedef Eigen::Matrix<Scalar, 3, 1> Point;
  /** \brief tangent vector type */
  typedef Eigen::Matrix<Scalar, DoF, 1> Tangent;
  /** \brief adjoint transformation type */
  typedef Eigen::Matrix<Scalar, DoF, DoF> Adjoint;

  /**
   * \brief Adjoint transformation
   *
   * This function return the adjoint transformation \f$ Ad \f$ of the
   * group instance \f$ A \f$  such that for all \f$ x \f$
   * it holds that \f$ \widehat{Ad_A\cdot x} = A\widehat{x}A^{-1} \f$
   * with \f$\ \widehat{\cdot} \f$ being the hat()-operator.
   *
   * For RxSO3, it simply returns the rotation matrix corresponding to
   * \f$ A \f$.
   */
  SOPHUS_FUNC Adjoint Adj() const {
    Adjoint res;
    res.setIdentity();
    res.template topLeftCorner<3, 3>() = rotationMatrix();
    return res;
  }

  /**
   * \returns copy of instance casted to NewScalarType
   */
  template <typename NewScalarType>
  SOPHUS_FUNC RxSO3Group<NewScalarType> cast() const {
    return RxSO3Group<NewScalarType>(
        quaternion().template cast<NewScalarType>());
  }

  /**
   * \returns pointer to internal data
   *
   * This provides direct read/write access to internal data. RxSO3 is
   * represented by an Eigen::Quaternion (four parameters).
   *
   * Note: The first three Scalars represent the imaginary parts, while the
   * forth Scalar represent the real part.
   */
  SOPHUS_FUNC Scalar* data() { return quaternion().coeffs().data(); }

  /**
   * \returns const pointer to internal data
   *
   * Const version of data().
   */
  SOPHUS_FUNC const Scalar* data() const {
    return quaternion().coeffs().data();
  }

  /**
   * \returns group inverse of instance
   */
  SOPHUS_FUNC RxSO3Group<Scalar> inverse() const {
    return RxSO3Group<Scalar>(quaternion().inverse());
  }

  /**
   * \brief Logarithmic map
   *
   * \returns tangent space representation (=rotation vector) of instance
   *
   * \see  log().
   */
  SOPHUS_FUNC Tangent log() const { return RxSO3Group<Scalar>::log(*this); }

  /**
   * \returns 3x3 matrix representation of instance
   *
   * For RxSO3, the matrix representation is a scaled orthogonal
   * matrix \f$ sR \f$ with \f$ det(sR)=s^3 \f$, thus a scaled rotation
   * matrix \f$ R \f$  with scale s.
   */
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

  /**
   * \brief Assignment operator
   */
  template <typename OtherDerived>
  SOPHUS_FUNC RxSO3GroupBase<Derived>& operator=(
      const RxSO3GroupBase<OtherDerived>& other) {
    quaternion() = other.quaternion();
    return *this;
  }

  /**
   * \brief Group multiplication
   * \see operator*=()
   */
  SOPHUS_FUNC RxSO3Group<Scalar> operator*(
      const RxSO3Group<Scalar>& other) const {
    RxSO3Group<Scalar> result(*this);
    result *= other;
    return result;
  }

  /**
   * \brief Group action on \f$ \mathbf{R}^3 \f$
   *
   * \param p point \f$p \in \mathbf{R}^3 \f$
   * \returns point \f$p' \in \mathbf{R}^3 \f$,
   *          rotated and scaled version of \f$p\f$
   *
   * This function rotates and scales a point \f$ p \f$ in  \f$ \mathbf{R}^3 \f$
   * by the RxSO3 transformation \f$sR\f$ (=rotation matrix)
   * : \f$ p' = sR\cdot p \f$.
   */
  SOPHUS_FUNC Point operator*(const Point& p) const {
    // Follows http://eigen.tuxfamily.org/bz/show_bug.cgi?id=459
    Scalar scale = quaternion().squaredNorm();
    Point two_vec_cross_p = quaternion().vec().cross(p);
    two_vec_cross_p += two_vec_cross_p;
    return scale * p + (quaternion().w() * two_vec_cross_p +
                        quaternion().vec().cross(two_vec_cross_p));
  }

  /**
   * \brief In-place group multiplication
   * \see operator*=()
   */
  SOPHUS_FUNC RxSO3GroupBase<Derived>& operator*=(
      const RxSO3Group<Scalar>& other) {
    using std::sqrt;

    quaternion() *= other.quaternion();
    Scalar scale = this->scale();
    if (scale < Constants<Scalar>::epsilon()) {
      SOPHUS_ENSURE(scale > 0, "Scale must be greater zero.");
      // Saturation to ensure class invariant.
      quaternion().normalize();
      quaternion().coeffs() *= sqrt(Constants<Scalar>::epsilon());
    }
    return *this;
  }

  /**
   * \brief Mutator of quaternion
   */
  SOPHUS_FUNC
  QuaternionReference quaternion() {
    return static_cast<Derived*>(this)->quaternion();
  }

  /**
   * \brief Accessor of quaternion
   */
  SOPHUS_FUNC
  ConstQuaternionReference quaternion() const {
    return static_cast<const Derived*>(this)->quaternion();
  }

  /**
   * \returns rotation matrix
   */
  SOPHUS_FUNC Transformation rotationMatrix() const {
    Eigen::Quaternion<Scalar> norm_quad = quaternion();
    norm_quad.normalize();
    return norm_quad.toRotationMatrix();
  }

  /**
   * \returns scale
   */
  SOPHUS_FUNC
  Scalar scale() const { return quaternion().squaredNorm(); }

  /**
   * \brief Setter of quaternion using rotation matrix, leaves scale untouched
   *
   * \param R a 3x3 rotation matrix
   * \pre       the 3x3 matrix should be orthogonal and have a determinant of 1
   */
  SOPHUS_FUNC void setRotationMatrix(const Transformation& R) {
    Scalar saved_scale = scale();
    quaternion() = R;
    quaternion() *= saved_scale;
  }

  /**
   * \brief Scale setter
   */
  SOPHUS_FUNC
  void setScale(const Scalar& scale) {
    using std::sqrt;
    quaternion().normalize();
    quaternion().coeffs() *= sqrt(scale);
  }

  /**
   * \brief Setter of quaternion using scaled rotation matrix
   *
   * \param sR a 3x3 scaled rotation matrix
   * \pre        the 3x3 matrix should be "scaled orthogonal"
   *             and have a positive determinant
   */
  SOPHUS_FUNC void setScaledRotationMatrix(const Transformation& sR) {
    Transformation squared_sR = sR * sR.transpose();
    Scalar squared_scale =
        static_cast<Scalar>(1. / 3.) *
        (squared_sR(0, 0) + squared_sR(1, 1) + squared_sR(2, 2));
    SOPHUS_ENSURE(squared_scale > Constants<Scalar>::epsilon() *
                                      Constants<Scalar>::epsilon(),
                  "Scale factor must be greater-equal epsilon.");
    Scalar scale = sqrt(squared_scale);
    quaternion() = sR / scale;
    quaternion().coeffs() *= sqrt(scale);
  }

  ////////////////////////////////////////////////////////////////////////////
  // public static functions
  ////////////////////////////////////////////////////////////////////////////

  /**
   * \param   b 4-vector representation of Lie algebra element
   * \returns   derivative of Lie bracket
   *
   * This function returns \f$ \frac{\partial}{\partial a} [a, b]_{rxso3} \f$
   * with \f$ [a, b]_{rxso3} \f$ being the lieBracket() of the Lie
   * algebra rxso3.
   *
   * \see lieBracket()
   */
  SOPHUS_FUNC static Adjoint d_lieBracketab_by_d_a(const Tangent& b) {
    Adjoint res;
    res.setZero();
    res.template topLeftCorner<3, 3>() =
        -SO3Group<Scalar>::hat(b.template head<3>());
    return res;
  }

  /**
   * \brief Group exponential
   *
   * \param a tangent space element
   *          (rotation vector \f$ \omega \f$ and logarithm of scale)
   * \returns corresponding element of the group RxSO3
   *
   * To be more specific, this function computes \f$ \exp(\widehat{a}) \f$
   * with \f$ \exp(\cdot) \f$ being the matrix exponential
   * and \f$ \widehat{\cdot} \f$ the hat()-operator of RxSO3.
   *
   * \see expAndTheta()
   * \see hat()
   * \see log()
   */
  SOPHUS_FUNC static RxSO3Group<Scalar> exp(const Tangent& a) {
    Scalar theta;
    return expAndTheta(a, &theta);
  }

  /**
   * \brief Group exponential and theta
   *
   * \param      a     tangent space element
   *                   (rotation vector \f$ \omega \f$ and logarithm of scale )
   * \param[out] theta angle of rotation \f$ \theta = |\omega| \f$
   * \returns          corresponding element of the group RxSO3
   *
   * \see exp() for details
   */
  SOPHUS_FUNC static RxSO3Group<Scalar> expAndTheta(const Tangent& a,
                                                    Scalar* theta) {
    using std::exp;
    using std::log;

    const Eigen::Matrix<Scalar, 3, 1>& omega = a.template head<3>();
    Scalar sigma = a[3];
    Scalar sqrt_scale = sqrt(exp(sigma));
    Eigen::Quaternion<Scalar> quat =
        SO3Group<Scalar>::expAndTheta(omega, theta).unit_quaternion();
    quat.coeffs() *= sqrt_scale;
    return RxSO3Group<Scalar>(quat);
  }

  /**
   * \brief Generators
   *
   * \pre \f$ i \in \{0,1,2,3\} \f$
   * \returns \f$ i \f$th generator \f$ G_i \f$ of RxSO3
   *
   * The infinitesimal generators of RxSO3
   * are \f$
   *        G_0 = \left( \begin{array}{ccc}
   *                          0&  0&  0& \\
   *                          0&  0& -1& \\
   *                          0&  1&  0&
   *                     \end{array} \right),
   *        G_1 = \left( \begin{array}{ccc}
   *                          0&  0&  1& \\
   *                          0&  0&  0& \\
   *                         -1&  0&  0&
   *                     \end{array} \right),
   *        G_2 = \left( \begin{array}{ccc}
   *                          0& -1&  0& \\
   *                          1&  0&  0& \\
   *                          0&  0&  0&
   *                     \end{array} \right),
   *        G_3 = \left( \begin{array}{ccc}
   *                          1&  0&  0& \\
   *                          0&  1&  0& \\
   *                          0&  0&  1&
   *                     \end{array} \right).
   * \f$
   * \see hat()
   */
  SOPHUS_FUNC static Transformation generator(int i) {
    SOPHUS_ENSURE(i >= 0 && i <= 3, "i should be in range [0,3].");
    Tangent e;
    e.setZero();
    e[i] = static_cast<Scalar>(1);
    return hat(e);
  }

  /**
   * \brief hat-operator
   *
   * \param a 4-vector representation of Lie algebra element
   * \returns 3x3-matrix representatin of Lie algebra element
   *
   * Formally, the hat-operator of RxSO3 is defined
   * as \f$ \widehat{\cdot}: \mathbf{R}^4 \rightarrow \mathbf{R}^{3\times 3},
   * \quad \widehat{a} = \sum_{i=0}^3 G_i a_i \f$
   * with \f$ G_i \f$ being the ith infinitesial generator().
   *
   * \see generator()
   * \see vee()
   */
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

  /**
   * \brief Lie bracket
   *
   * \param a 4-vector representation of Lie algebra element
   * \param b 4-vector representation of Lie algebra element
   * \returns 4-vector representation of Lie algebra element
   *
   * It computes the bracket of RxSO3. To be more specific, it
   * computes \f$ [a, 2]_{rxso3}
   * := [\widehat{a}, \widehat{b}]^\vee \f$
   * with \f$ [A,B] = AB-BA \f$ being the matrix
   * commutator, \f$ \widehat{\cdot} \f$ the
   * hat()-operator and \f$ (\cdot)^\vee \f$ the vee()-operator of RxSO3.
   *
   * \see hat()
   * \see vee()
   */
  SOPHUS_FUNC static Tangent lieBracket(const Tangent& a, const Tangent& b) {
    const Eigen::Matrix<Scalar, 3, 1>& omega1 = a.template head<3>();
    const Eigen::Matrix<Scalar, 3, 1>& omega2 = b.template head<3>();
    Eigen::Matrix<Scalar, 4, 1> res;
    res.template head<3>() = omega1.cross(omega2);
    res[3] = static_cast<Scalar>(0);
    return res;
  }

  /**
   * \brief Logarithmic map
   *
   * \param other element of the group RxSO3
   * \returns     corresponding tangent space element
   *              (rotation vector \f$ \omega \f$ and logarithm of scale)
   *
   * Computes the logarithmic, the inverse of the group exponential.
   * To be specific, this function computes \f$ \log({\cdot})^\vee \f$
   * with \f$ \vee(\cdot) \f$ being the matrix logarithm
   * and \f$ \vee{\cdot} \f$ the vee()-operator of RxSO3.
   *
   * \see exp()
   * \see logAndTheta()
   * \see vee()
   */
  SOPHUS_FUNC static Tangent log(const RxSO3Group<Scalar>& other) {
    Scalar theta;
    return logAndTheta(other, &theta);
  }

  /**
   * \brief Logarithmic map and theta
   *
   * \param      other element of the group RxSO3
   * \param[out] theta angle of rotation \f$ \theta = |\omega| \f$
   * \returns          corresponding tangent space element
   *                   (rotation vector \f$ \omega \f$ and logarithm of scale)
   *
   * \see log() for details
   */
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

  /**
   * \brief vee-operator
   *
   * \param Omega 3x3-matrix representation of Lie algebra element
   * \returns     4-vector representatin of Lie algebra element
   *
   * This is the inverse of the hat()-operator.
   *
   * \see hat()
   */
  SOPHUS_FUNC static Tangent vee(const Transformation& Omega) {
    return Tangent(static_cast<Scalar>(0.5) * (Omega(2, 1) - Omega(1, 2)),
                   static_cast<Scalar>(0.5) * (Omega(0, 2) - Omega(2, 0)),
                   static_cast<Scalar>(0.5) * (Omega(1, 0) - Omega(0, 1)),
                   static_cast<Scalar>(1. / 3.) *
                       (Omega(0, 0) + Omega(1, 1) + Omega(2, 2)));
  }
};

/**
 * \brief RxSO3 default type - Constructors and default storage for RxSO3 Type
 */
template <typename _Scalar, int _Options>
class RxSO3Group : public RxSO3GroupBase<RxSO3Group<_Scalar, _Options>> {
  typedef RxSO3GroupBase<RxSO3Group<_Scalar, _Options>> Base;

 public:
  /** \brief scalar type */
  typedef typename Eigen::internal::traits<SO3Group<_Scalar, _Options>>::Scalar
      Scalar;
  /** \brief quaternion reference type */
  typedef typename Eigen::internal::traits<
      SO3Group<_Scalar, _Options>>::QuaternionType& QuaternionReference;
  /** \brief quaternion const reference type */
  typedef const typename Eigen::internal::traits<
      SO3Group<_Scalar, _Options>>::QuaternionType& ConstQuaternionReference;

  /** \brief group transfomation type */
  typedef typename Base::Transformation Transformation;
  /** \brief point type */
  typedef typename Base::Point Point;
  /** \brief tangent vector type */
  typedef typename Base::Tangent Tangent;
  /** \brief adjoint transformation type */
  typedef typename Base::Adjoint Adjoint;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * \brief Default constructor
   *
   * Initialize Eigen::Quaternion to identity rotation and scale.
   */
  SOPHUS_FUNC RxSO3Group()
      : quaternion_(static_cast<Scalar>(1), static_cast<Scalar>(0),
                    static_cast<Scalar>(0), static_cast<Scalar>(0)) {}

  /**
   * \brief Copy constructor
   */
  template <typename OtherDerived>
  SOPHUS_FUNC RxSO3Group(const RxSO3GroupBase<OtherDerived>& other)
      : quaternion_(other.quaternion()) {}

  /**
   * \brief Constructor from scaled rotation matrix
   *
   * \pre matrix need to be "scaled orthogonal" with positive determinant
   */
  SOPHUS_FUNC explicit RxSO3Group(const Transformation& sR) {
    this->setScaledRotationMatrix(sR);
  }

  /**
   * \brief Constructor from scale factor and rotation matrix
   *
   * \pre rotation matrix need to be orthogonal with determinant of 1
   * \pre scale need to be not zero
   */
  SOPHUS_FUNC RxSO3Group(const Scalar& scale, const Transformation& R)
      : quaternion_(R) {
    SOPHUS_ENSURE(scale >= Constants<Scalar>::epsilon(),
                  "Scale factor must be greater-equal epsilon.");
    quaternion_.normalize();
    quaternion_.coeffs() *= scale;
  }

  /**
   * \brief Constructor from scale factor and SO3
   *
   * \pre scale need to be not zero
   */
  SOPHUS_FUNC RxSO3Group(const Scalar& scale, const SO3Group<Scalar>& so3)
      : quaternion_(so3.unit_quaternion()) {
    SOPHUS_ENSURE(scale >= Constants<Scalar>::epsilon(),
                  "Scale factor must be greater-equal epsilon.");
    quaternion_.normalize();
    quaternion_.coeffs() *= scale;
  }

  /**
   * \brief Constructor from quaternion
   *
   * \pre quaternion must not be zero
   */
  SOPHUS_FUNC explicit RxSO3Group(const Eigen::Quaternion<Scalar>& quat)
      : quaternion_(quat) {
    SOPHUS_ENSURE(quaternion_.squaredNorm() > Constants<Scalar>::epsilon(),
                  "Scale factor must be greater-equal epsilon.");
  }

  /**
   * \brief Mutator of quaternion
   */
  SOPHUS_FUNC
  QuaternionReference quaternion() { return quaternion_; }

  /**
   * \brief Accessor of quaternion
   */
  SOPHUS_FUNC
  ConstQuaternionReference quaternion() const { return quaternion_; }

 protected:
  Eigen::Quaternion<Scalar> quaternion_;
};

}  // end namespace

namespace Eigen {
/**
 * \brief Specialisation of Eigen::Map for RxSO3GroupBase
 *
 * Allows us to wrap RxSO3 Objects around POD array
 * (e.g. external c style quaternion)
 */
template <typename _Scalar, int _Options>
class Map<Sophus::RxSO3Group<_Scalar>, _Options>
    : public Sophus::RxSO3GroupBase<
          Map<Sophus::RxSO3Group<_Scalar>, _Options>> {
  typedef Sophus::RxSO3GroupBase<Map<Sophus::RxSO3Group<_Scalar>, _Options>>
      Base;

 public:
  /** \brief scalar type */
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  /** \brief quaternion reference type */
  typedef typename Eigen::internal::traits<Map>::QuaternionType&
      QuaternionReference;
  /** \brief quaternion const reference type */
  typedef const typename Eigen::internal::traits<Map>::QuaternionType&
      ConstQuaternionReference;

  /** \brief group transfomation type */
  typedef typename Base::Transformation Transformation;
  /** \brief point type */
  typedef typename Base::Point Point;
  /** \brief tangent vector type */
  typedef typename Base::Tangent Tangent;
  /** \brief adjoint transformation type */
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  Map(Scalar* coeffs) : quaternion_(coeffs) {}

  /**
   * \brief Mutator of quaternion
   */
  SOPHUS_FUNC
  QuaternionReference quaternion() { return quaternion_; }

  /**
   * \brief Accessor of quaternion
   */
  SOPHUS_FUNC
  ConstQuaternionReference quaternion() const { return quaternion_; }

 protected:
  Map<Eigen::Quaternion<Scalar>, _Options> quaternion_;
};

/**
 * \brief Specialisation of Eigen::Map for const RxSO3GroupBase
 *
 * Allows us to wrap RxSO3 Objects around POD array
 * (e.g. external c style quaternion)
 */
template <typename _Scalar, int _Options>
class Map<const Sophus::RxSO3Group<_Scalar>, _Options>
    : public Sophus::RxSO3GroupBase<
          Map<const Sophus::RxSO3Group<_Scalar>, _Options>> {
  typedef Sophus::RxSO3GroupBase<
      Map<const Sophus::RxSO3Group<_Scalar>, _Options>>
      Base;

 public:
  /** \brief scalar type */
  typedef typename Eigen::internal::traits<Map>::Scalar Scalar;
  /** \brief quaternion const reference type */
  typedef const typename Eigen::internal::traits<Map>::QuaternionType&
      ConstQuaternionReference;

  /** \brief group transfomation type */
  typedef typename Base::Transformation Transformation;
  /** \brief point type */
  typedef typename Base::Point Point;
  /** \brief tangent vector type */
  typedef typename Base::Tangent Tangent;
  /** \brief adjoint transformation type */
  typedef typename Base::Adjoint Adjoint;

  EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Map)
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC
  Map(const Scalar* coeffs) : quaternion_(coeffs) {}

  /**
   * \brief Accessor of unit quaternion
   *
   * No direct write access is given to ensure the quaternion stays normalized.
   */
  SOPHUS_FUNC
  ConstQuaternionReference quaternion() const { return quaternion_; }

 protected:
  const Map<const Eigen::Quaternion<Scalar>, _Options> quaternion_;
};
}

#endif  // SOPHUS_RXSO3_HPP
