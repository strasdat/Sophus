// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once
#include "sophus/concepts/lie_group.h"
#include "sophus/manifold/quaternion.h"
#include "sophus/manifold/unit_vector.h"

namespace sophus {
namespace lie {

template <class TScalar>
class Rotation3Impl {
 public:
  using Scalar = TScalar;
  using Quaternion = QuaternionImpl<TScalar>;

  static bool constexpr kIsOriginPreserving = true;
  static bool constexpr kIsAxisDirectionPreserving = false;
  static bool constexpr kIsDirectionVectorPreserving = false;
  static bool constexpr kIsShapePreserving = true;
  static bool constexpr kIisSizePreserving = true;
  static bool constexpr kIisParallelLinePreserving = true;

  static int const kDof = 3;
  static int const kNumParams = 4;
  static int const kPointDim = 3;
  static int const kAmbientDim = 3;

  using Tangent = Eigen::Vector<Scalar, kDof>;
  using Params = Eigen::Vector<Scalar, kNumParams>;
  using Point = Eigen::Vector<Scalar, kPointDim>;

  template <class TCompatibleScalar>
  using ScalarReturn = typename Eigen::
      ScalarBinaryOpTraits<Scalar, TCompatibleScalar>::ReturnType;

  template <class TCompatibleScalar>
  using ParamsReturn =
      Eigen::Vector<ScalarReturn<TCompatibleScalar>, kNumParams>;

  template <class TCompatibleScalar>
  using PointReturn = Eigen::Vector<ScalarReturn<TCompatibleScalar>, kPointDim>;

  template <class TCompatibleScalar>
  using UnitVectorReturn =
      UnitVector<ScalarReturn<TCompatibleScalar>, kPointDim>;

  // constructors and factories

  static auto identityParams() -> Params {
    return Eigen::Vector<Scalar, 4>(
        Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(1.0));
  }

  static auto areParamsValid(Params const& unit_quaternion)
      -> sophus::Expected<Success> {
    static const Scalar kThr = kEpsilonSqrt<Scalar>;
    const Scalar squared_norm = unit_quaternion.squaredNorm();
    using std::abs;
    if (!(abs(squared_norm - 1.0) <= kThr)) {
      return SOPHUS_UNEXPECTED(
          "quaternion number (({}), {}) is not unit length.\n"
          "Squared norm: {}, thr: {}",
          unit_quaternion.template head<3>(),
          unit_quaternion[3],
          squared_norm,
          kThr);
    }
    return sophus::Expected<Success>{};
  }

  static auto hasShortestPathAmbiguity(Params const& foo_from_bar) -> bool {
    using std::abs;
    TScalar angle = Rotation3Impl::log(foo_from_bar).norm();
    TScalar const k_pi = kPi<TScalar>;  // NOLINT
    return abs(angle - k_pi) / (angle + k_pi) < kEpsilon<TScalar>;
  }

  // Manifold / Lie Group concepts

  static auto exp(Tangent const& omega) -> Params {
    using std::abs;
    using std::cos;
    using std::sin;
    using std::sqrt;
    Scalar theta;
    Scalar theta_sq = omega.squaredNorm();

    Scalar imag_factor;
    Scalar real_factor;
    if (theta_sq < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      theta = Scalar(0);
      Scalar theta_po4 = theta_sq * theta_sq;
      imag_factor = Scalar(0.5) - Scalar(1.0 / 48.0) * theta_sq +
                    Scalar(1.0 / 3840.0) * theta_po4;
      real_factor = Scalar(1) - Scalar(1.0 / 8.0) * theta_sq +
                    Scalar(1.0 / 384.0) * theta_po4;
    } else {
      theta = sqrt(theta_sq);
      Scalar half_theta = Scalar(0.5) * (theta);
      Scalar sin_half_theta = sin(half_theta);
      imag_factor = sin_half_theta / (theta);
      real_factor = cos(half_theta);
    }

    return Params(
        imag_factor * omega.x(),
        imag_factor * omega.y(),
        imag_factor * omega.z(),
        real_factor);
  }

  static auto log(Params const& unit_quaternion) -> Tangent {
    using std::abs;
    using std::atan2;
    using std::sqrt;
    Eigen::Vector3<Scalar> ivec = unit_quaternion.template head<3>();

    Scalar squared_n = ivec.squaredNorm();
    Scalar w = unit_quaternion.w();

    Scalar two_atan_nbyw_by_n;

    // Atan-based log thanks to
    //
    // C. Hertzberg et al.:
    // "Integrating Generic Sensor Fusion Algorithms with Sound State
    // Representation through Encapsulation of Manifolds"
    // Information Fusion, 2011
    if (squared_n < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      // If quaternion is normalized and n=0, then w should be 1;
      // w=0 should never happen here!
      SOPHUS_ASSERT(
          abs(w) >= kEpsilon<Scalar>,
          "Quaternion ({}) should be normalized!",
          unit_quaternion);
      Scalar squared_w = w * w;
      two_atan_nbyw_by_n =
          Scalar(2) / w - Scalar(2.0 / 3.0) * (squared_n) / (w * squared_w);
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
    }
    return two_atan_nbyw_by_n * ivec;
  }

  static auto hat(Tangent const& omega)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> mat_omega;
    // clang-format off
    mat_omega <<
        Scalar(0), -omega(2),  omega(1),
         omega(2), Scalar(0), -omega(0),
        -omega(1),  omega(0), Scalar(0);
    // clang-format on
    return mat_omega;
  }

  static auto vee(
      Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat_omega)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return Eigen::Matrix<Scalar, kDof, 1>(
        mat_omega(2, 1), mat_omega(0, 2), mat_omega(1, 0));
  }

  // group operations

  static auto inverse(Params const& unit_quat) -> Params {
    return Quaternion::conjugate(unit_quat);
  }

  template <class TCompatibleScalar>
  static auto multiplication(
      Params const& lhs_params,
      Eigen::Vector<TCompatibleScalar, 4> const& rhs_params)
      -> ParamsReturn<TCompatibleScalar> {
    ParamsReturn<TCompatibleScalar> result =
        Quaternion::multiplication(lhs_params, rhs_params);
    using SReturn = ScalarReturn<TCompatibleScalar>;
    SReturn const squared_norm = result.squaredNorm();

    // We can assume that the squared-norm is close to 1 since we deal with a
    // unit complex number. Due to numerical precision issues, there might
    // be a small drift after pose concatenation. Hence, we need to
    // the complex number here.
    // Since squared-norm is close to 1, we do not need to calculate the costly
    // square-root, but can use an approximation around 1 (see
    // http://stackoverflow.com/a/12934750 for details).
    if (squared_norm != SReturn(1.0)) {
      SReturn const scale = SReturn(2.0) / (SReturn(1.0) + squared_norm);
      return scale * result;
    }
    return result;
  }

  static auto adj(Params const& params) -> Eigen::Matrix<Scalar, kDof, kDof> {
    return matrix(params);
  }

  // Point actions

  template <class TCompatibleScalar>
  static auto action(
      Params const& unit_quat,
      Eigen::Vector<TCompatibleScalar, kPointDim> const& point)
      -> PointReturn<TCompatibleScalar> {
    Eigen::Vector3<Scalar> ivec = unit_quat.template head<3>();

    PointReturn<TCompatibleScalar> uv = ivec.cross(point);
    uv += uv;
    return point + unit_quat.w() * uv + ivec.cross(uv);
  }

  template <class TCompatibleScalar>
  static auto action(
      Params const& unit_quat,
      UnitVector<TCompatibleScalar, 3> const& direction_vector)
      -> UnitVectorReturn<TCompatibleScalar> {
    // TODO: Implement normalization using expansion around 1 as done for
    // ::multiplication to avoid possibly costly call to std::sqrt.
    return UnitVectorReturn<TCompatibleScalar>::fromVectorAndNormalize(
        action(unit_quat, direction_vector.params()));
  }

  static auto toAmbient(Point const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }

  // matrices

  static auto compactMatrix(Params const& unit_quat)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    Eigen::Vector3<Scalar> ivec = unit_quat.template head<3>();
    Scalar real = unit_quat.w();
    return Eigen::Matrix<Scalar, 3, 3>{
        {Scalar(1.0 - 2.0 * (ivec[1] * ivec[1]) - 2.0 * (ivec[2] * ivec[2])),
         Scalar(2.0 * ivec[0] * ivec[1] - 2.0 * ivec[2] * real),
         Scalar(2.0 * ivec[0] * ivec[2] + 2.0 * ivec[1] * real)},
        {
            Scalar(2.0 * ivec[0] * ivec[1] + 2.0 * ivec[2] * real),
            Scalar(1.0 - 2.0 * (ivec[0] * ivec[0]) - 2.0 * (ivec[2] * ivec[2])),
            Scalar(2.0 * ivec[1] * ivec[2] - 2.0 * ivec[0] * real),
        },
        {Scalar(2.0 * ivec[0] * ivec[2] - 2.0 * ivec[1] * real),
         Scalar(2.0 * ivec[1] * ivec[2] + 2.0 * ivec[0] * real),
         Scalar(1.0 - 2.0 * (ivec[0] * ivec[0]) - 2.0 * (ivec[1] * ivec[1]))}};
  }

  static auto matrix(Params const& unit_quat)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return compactMatrix(unit_quat);
  }

  // Sub-group concepts
  static auto matV(Params const& params, Tangent const& omega)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    using std::cos;
    using std::sin;
    using std::sqrt;

    Scalar const theta_sq = omega.squaredNorm();
    Eigen::Matrix3<Scalar> const mat_omega = hat(omega);
    Eigen::Matrix3<Scalar> const mat_omega_sq = mat_omega * mat_omega;
    Eigen::Matrix3<Scalar> v;

    if (theta_sq < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      v = Eigen::Matrix3<Scalar>::Identity() + Scalar(0.5) * mat_omega;
    } else {
      Scalar theta = sqrt(theta_sq);
      v = Eigen::Matrix3<Scalar>::Identity() +
          (Scalar(1) - cos(theta)) / theta_sq * mat_omega +
          (theta - sin(theta)) / (theta_sq * theta) * mat_omega_sq;
    }
    return v;
  }

  static auto matVInverse(Params const& /*unused*/, Tangent const& omega)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    using std::cos;
    using std::sin;
    using std::sqrt;
    Scalar const theta_sq = omega.squaredNorm();
    Eigen::Matrix3<Scalar> const mat_omega = hat(omega);

    Eigen::Matrix3<Scalar> v_inv;
    if (theta_sq < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      v_inv = Eigen::Matrix3<Scalar>::Identity() - Scalar(0.5) * mat_omega +
              Scalar(1. / 12.) * (mat_omega * mat_omega);

    } else {
      Scalar const theta = sqrt(theta_sq);
      Scalar const half_theta = Scalar(0.5) * theta;

      v_inv = Eigen::Matrix3<Scalar>::Identity() - Scalar(0.5) * mat_omega +
              (Scalar(1) -
               Scalar(0.5) * theta * cos(half_theta) / sin(half_theta)) /
                  (theta * theta) * (mat_omega * mat_omega);
    }
    return v_inv;
  }

  static auto adjOfTranslation(Params const& params, Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return hat(point) * matrix(params);
  }

  static auto adOfTranslation(Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return hat(point);
  }

  // derivatives
  static auto ad(Tangent const& omega) -> Eigen::Matrix<Scalar, kDof, kDof> {
    return hat(omega);
  }

  static auto dxExpX(Tangent const& omega)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
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

    Eigen::Matrix<Scalar, 4, 3> j;

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

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, 4, 3> j;
    // clang-format off
    j <<  Scalar(0.5),   Scalar(0),   Scalar(0),
            Scalar(0), Scalar(0.5),   Scalar(0),
            Scalar(0),   Scalar(0), Scalar(0.5),
            Scalar(0),   Scalar(0),   Scalar(0);
    // clang-format on
    return j;
  }

  static auto dxExpXTimesPointAt0(Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return hat(-point);
  }

  static auto dxThisMulExpXAt0(Params const& unit_quat)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, 4, 3> j;
    Scalar const c0 = Scalar(0.5) * unit_quat.w();
    Scalar const c1 = Scalar(0.5) * unit_quat.z();
    Scalar const c2 = -c1;
    Scalar const c3 = Scalar(0.5) * unit_quat.y();
    Scalar const c4 = Scalar(0.5) * unit_quat.x();
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

  static auto dxLogThisInvTimesXAtThis(Params const& q)
      -> Eigen::Matrix<Scalar, kDof, kNumParams> {
    Eigen::Matrix<Scalar, 3, 4> j;
    // clang-format off
    j << q.w(),  q.z(), -q.y(), -q.x(),
        -q.z(),  q.w(),  q.x(), -q.y(),
         q.y(), -q.x(),  q.w(), -q.z();
    // clang-format on
    return j * Scalar(2.);
  }

  // for tests

  static auto tangentExamples() -> std::vector<Tangent> {
    Scalar o(0);
    Scalar i(1);
    return std::vector<Tangent>({
        Tangent(o, o, o),
        Tangent(i, o, o),
        Tangent(o, i, o),
        Tangent{Scalar(0.5) * kPi<Scalar>, Scalar(0.5) * kPi<Scalar>, o},
        Tangent{-i, i, o},
        Tangent{Scalar(20), -i, o},
        Tangent{Scalar(30), Scalar(5), -i},
        Tangent{i, i, Scalar(4)},
        Tangent{i, Scalar(-3), Scalar(0.5)},
        Tangent{Scalar(-5), Scalar(-6), Scalar(7)},
    });
  }

  static auto paramsExamples() -> std::vector<Params> {
    using Point = Point;
    return std::vector<Params>(
        {Params(Scalar(0.1e-11), Scalar(0.), Scalar(1.), Scalar(0.)),
         Params(Scalar(-1), Scalar(0.00001), Scalar(0.0), Scalar(0.0)),
         exp(Point(Scalar(0.2), Scalar(0.5), Scalar(0.0))),
         exp(Point(Scalar(0.2), Scalar(0.5), Scalar(-1.0))),
         exp(Point(Scalar(0.), Scalar(0.), Scalar(0.))),
         exp(Point(Scalar(0.), Scalar(0.), Scalar(0.00001))),
         exp(Point(kPi<Scalar>, Scalar(0), Scalar(0))),
         multiplication(
             multiplication(
                 exp(Point(Scalar(0.2), Scalar(0.5), Scalar(0.0))),
                 exp(Point(kPi<Scalar>, Scalar(0), Scalar(0)))),
             exp(Point(kPi<Scalar>, Scalar(0), Scalar(0)))),
         multiplication(
             multiplication(
                 exp(Point(Scalar(0.3), Scalar(0.5), Scalar(0.1))),
                 exp(Point(kPi<Scalar>, Scalar(0), Scalar(0)))),
             exp(Point(Scalar(-0.3), Scalar(-0.5), Scalar(-0.1))))});
  }

  static auto invalidParamsExamples() -> std::vector<Params> {
    return std::vector<Params>({
        Params::Zero(),
        Params::Ones(),
        2.0 * Params::UnitX(),
    });
  }
};

}  // namespace lie
}  // namespace sophus
