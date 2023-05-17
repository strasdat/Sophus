// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once
#include "sophus/concepts/lie_group.h"
#include "sophus/lie/impl/rotation3.h"
#include "sophus/lie/impl/sim_mat_w.h"
#include "sophus/manifold/quaternion.h"
#include "sophus/manifold/unit_vector.h"

namespace sophus {
namespace lie {

template <class TScalar>
class SpiralSimilarity3Impl {
 public:
  using Scalar = TScalar;
  using Quaternion = QuaternionImpl<TScalar>;

  static bool constexpr kIsOriginPreserving = true;
  static bool constexpr kIsAxisDirectionPreserving = false;
  static bool constexpr kIsDirectionVectorPreserving = false;
  static bool constexpr kIsShapePreserving = false;
  static bool constexpr kIisSizePreserving = true;
  static bool constexpr kIisParallelLinePreserving = true;

  static int const kDof = 4;
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

  static auto areParamsValid(Params const& non_zero_quat)
      -> sophus::Expected<Success> {
    static const Scalar kThr = kEpsilon<Scalar> * kEpsilon<Scalar>;
    const Scalar squared_norm = non_zero_quat.squaredNorm();
    using std::abs;
    if (!(squared_norm > kThr || squared_norm < 1.0 / kThr)) {
      return SOPHUS_UNEXPECTED(
          "complex number ({}, {}) is too large or too small.\n"
          "Squared norm: {}, thr: {}",
          non_zero_quat[0],
          non_zero_quat[1],
          squared_norm,
          kThr);
    }
    return sophus::Expected<Success>{};
  }

  static auto hasShortestPathAmbiguity(Params const& non_zero_quat) -> bool {
    return Rotation3Impl<Scalar>::hasShortestPathAmbiguity(
        non_zero_quat.normalized());
  }

  // Manifold / Lie Group concepts

  static auto exp(Tangent const& omega_logscale) -> Params {
    using std::exp;
    using std::max;
    using std::min;
    using std::sqrt;

    Eigen::Vector3<Scalar> const vec_omega = omega_logscale.template head<3>();
    Scalar scale = exp(omega_logscale[3]);
    // Ensure that scale-factor contraint is always valid
    scale = max(scale, kEpsilonPlus<Scalar>);
    scale = min(scale, Scalar(1.) / kEpsilonPlus<Scalar>);
    Scalar sqrt_scale = sqrt(scale);
    Params quat = Rotation3Impl<Scalar>::exp(vec_omega);
    quat *= sqrt_scale;
    return quat;
  }

  static auto log(Params const& non_zero_quat) -> Tangent {
    using std::log;

    Scalar scale = non_zero_quat.squaredNorm();
    Tangent omega_and_logscale;
    omega_and_logscale.template head<3>() =
        Rotation3Impl<Scalar>::log(non_zero_quat.normalized());
    omega_and_logscale[3] = log(scale);
    return omega_and_logscale;
  }

  static auto hat(Tangent const& omega_logscale)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return Eigen::Matrix<Scalar, 3, 3>{
        {+omega_logscale(3), -omega_logscale(2), +omega_logscale(1)},
        {+omega_logscale(2), +omega_logscale(3), -omega_logscale(0)},
        {-omega_logscale(1), +omega_logscale(0), +omega_logscale(3)}};
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return Eigen::Matrix<Scalar, kDof, 1>{
        mat(2, 1), mat(0, 2), mat(1, 0), mat(0, 0)};
  }

  // group operations

  static auto inverse(Params const& non_zero_quat) -> Params {
    Scalar squared_scale = non_zero_quat.norm();
    return (1.0 / squared_scale) *
           QuaternionImpl<Scalar>::conjugate(non_zero_quat.normalized());
  }

  template <class TCompatibleScalar>
  static auto multiplication(
      Params const& lhs_params,
      Eigen::Vector<TCompatibleScalar, kNumParams> const& rhs_params)
      -> ParamsReturn<TCompatibleScalar> {
    ParamsReturn<TCompatibleScalar> result =
        QuaternionImpl<Scalar>::multiplication(lhs_params, rhs_params);
    using R = ScalarReturn<TCompatibleScalar>;
    R const squared_scale = result.squaredNorm();

    if (squared_scale < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      /// Saturation to ensure class invariant.
      result.normalize();
      result *= kEpsilonPlus<R>;
    }
    if (squared_scale > Scalar(1.) / (kEpsilon<Scalar> * kEpsilon<Scalar>)) {
      /// Saturation to ensure class invariant.
      result.normalize();
      result /= kEpsilonPlus<R>;
    }
    return result;
  }

  // Point actions
  template <class TCompatibleScalar>
  static auto action(
      Params const& non_zero_quat,
      Eigen::Vector<TCompatibleScalar, kPointDim> const& point)
      -> PointReturn<TCompatibleScalar> {
    return matrix(non_zero_quat) * point;
  }

  template <class TCompatibleScalar>
  static auto action(
      Params const& non_zero_quat,
      UnitVector<TCompatibleScalar, kPointDim> const& direction_vector)
      -> UnitVectorReturn<TCompatibleScalar> {
    return UnitVectorReturn<TCompatibleScalar>::fromParams(
        Rotation3Impl<Scalar>::matrix(non_zero_quat.normalized()) *
        direction_vector.params());
  }

  static auto toAmbient(Point const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }

  static auto adj(Params const& non_zero_quat)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    Eigen::Matrix<Scalar, kDof, kDof> mat;
    mat.setIdentity();
    mat.template topLeftCorner<3, 3>() =
        Rotation3Impl<Scalar>::matrix(non_zero_quat.normalized());
    return mat;
  }

  // matrices

  static auto compactMatrix(Params const& non_zero_quat)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    Eigen::Matrix<Scalar, kPointDim, kPointDim> s_r;

    Scalar const vx_sq = non_zero_quat.x() * non_zero_quat.x();
    Scalar const vy_sq = non_zero_quat.y() * non_zero_quat.y();
    Scalar const vz_sq = non_zero_quat.z() * non_zero_quat.z();
    Scalar const w_sq = non_zero_quat.w() * non_zero_quat.w();
    Scalar const two_vx = Scalar(2) * non_zero_quat.x();
    Scalar const two_vy = Scalar(2) * non_zero_quat.y();
    Scalar const two_vz = Scalar(2) * non_zero_quat.z();
    Scalar const two_vx_vy = two_vx * non_zero_quat.y();
    Scalar const two_vx_vz = two_vx * non_zero_quat.z();
    Scalar const two_vx_w = two_vx * non_zero_quat.w();
    Scalar const two_vy_vz = two_vy * non_zero_quat.z();
    Scalar const two_vy_w = two_vy * non_zero_quat.w();
    Scalar const two_vz_w = two_vz * non_zero_quat.w();

    s_r(0, 0) = vx_sq - vy_sq - vz_sq + w_sq;
    s_r(1, 0) = two_vx_vy + two_vz_w;
    s_r(2, 0) = two_vx_vz - two_vy_w;

    s_r(0, 1) = two_vx_vy - two_vz_w;
    s_r(1, 1) = -vx_sq + vy_sq - vz_sq + w_sq;
    s_r(2, 1) = two_vx_w + two_vy_vz;

    s_r(0, 2) = two_vx_vz + two_vy_w;
    s_r(1, 2) = -two_vx_w + two_vy_vz;
    s_r(2, 2) = -vx_sq - vy_sq + vz_sq + w_sq;
    return s_r;
  }

  static auto matrix(Params const& non_zero_quat)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return compactMatrix(non_zero_quat);
  }

  // Sub-group concepts
  static auto matV(Params const& /*unused*/, Tangent const& angle_logscale)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    Eigen::Matrix<Scalar, 3, 1> omega = angle_logscale.template head<3>();
    Eigen::Matrix<Scalar, 3, 3> mat_omega = Rotation3Impl<Scalar>::hat(omega);
    Scalar theta = omega.norm();
    Eigen::Matrix3<Scalar> const w =
        details::calcW<Scalar, 3>(mat_omega, theta, angle_logscale[3]);
    return w;
  }

  static auto matVInverse(
      Params const& non_zero_quat, Tangent const& angle_logscale)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    Eigen::Matrix<Scalar, 3, 1> omega = angle_logscale.template head<3>();
    Eigen::Matrix<Scalar, 3, 3> mat_omega = Rotation3Impl<Scalar>::hat(omega);
    Scalar theta = omega.norm();
    return details::calcWInv<Scalar, 3>(
        mat_omega, theta, angle_logscale[3], non_zero_quat.squaredNorm());
  }

  static auto adjOfTranslation(Params const& quat, Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, 3, 4> tr_adj;
    tr_adj.template topLeftCorner<3, 3>() =
        Rotation3Impl<Scalar>::hat(point) *
        Rotation3Impl<Scalar>::matrix(quat.normalized());
    tr_adj.template topRightCorner<3, 1>() = -point;
    return tr_adj;
  }

  static auto adOfTranslation(Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, 3, 4> tr_ad;
    tr_ad.template leftCols<3>() = Rotation3Impl<Scalar>::hat(point);
    tr_ad.col(3) = -point;
    return tr_ad;
  }

  // derivatives

  static auto ad(Tangent const& tangent) -> Eigen::Matrix<Scalar, kDof, kDof> {
    Eigen::Matrix<Scalar, kDof, kDof> mat;
    mat.setZero();
    mat.template topLeftCorner<3, 3>() =
        Rotation3Impl<Scalar>::hat(tangent.template head<3>());
    return mat;
  }

  static auto dxExpX(Tangent const& a)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    using std::exp;
    using std::sqrt;
    Eigen::Matrix<Scalar, 4, 4> j;
    Eigen::Vector3<Scalar> const omega = a.template head<3>();
    Scalar const sigma = a[3];
    Params quat = Rotation3Impl<Scalar>::exp(omega);
    Scalar const scale = sqrt(exp(sigma));
    Scalar const scale_half = scale * Scalar(0.5);

    j.template block<4, 3>(0, 0) = Rotation3Impl<Scalar>::dxExpX(omega) * scale;
    j.col(3) = a * scale_half;
    return j;
  }

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    static Scalar const kH(0.5);
    return kH * Eigen::Matrix<Scalar, 4, 4>::Identity();
  }

  static auto dxExpXTimesPointAt0(Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, 3, 4> j;
    j << Rotation3Impl<Scalar>::hat(-point), point;
    return j;
  }

  static auto dxThisMulExpXAt0(Params const& non_zero_quat)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, 4, 4> j;
    j.col(3) = non_zero_quat * Scalar(0.5);
    Scalar const c0 = Scalar(0.5) * non_zero_quat.w();
    Scalar const c1 = Scalar(0.5) * non_zero_quat.z();
    Scalar const c2 = -c1;
    Scalar const c3 = Scalar(0.5) * non_zero_quat.y();
    Scalar const c4 = Scalar(0.5) * non_zero_quat.x();
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
    Eigen::Matrix<Scalar, 4, 4> j;
    // clang-format off
    j << q.w(),  q.z(), -q.y(), -q.x(),
        -q.z(),  q.w(),  q.x(), -q.y(),
         q.y(), -q.x(),  q.w(), -q.z(),
         q.x(),  q.y(),  q.z(),  q.w();
    // clang-format on
    const Scalar scaler = Scalar(2.) / q.squaredNorm();
    return j * scaler;
  }

  // for tests

  static auto tangentExamples() -> std::vector<Tangent> {
    return std::vector<Tangent>({
        Tangent{Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0)},
        Tangent{Scalar(1.0), Scalar(0.0), Scalar(0.0), Scalar(0.0)},
        Tangent{Scalar(1.0), Scalar(0.0), Scalar(0.0), Scalar(0.1)},
        Tangent{Scalar(0.0), Scalar(1.0), Scalar(0.0), Scalar(0.1)},
        Tangent{Scalar(0.00001), Scalar(0.00001), Scalar(0.0), Scalar(0.3)},
        Tangent{
            Scalar(0.5 * kPi<Scalar>), Scalar(0.9), Scalar(0.0), Scalar(0.0)},
        Tangent{
            Scalar(0.0),
            Scalar(0.0),
            Scalar(0.5 * kPi<Scalar> + 0.00001),
            Scalar(0.2)},
    });
  }

  static auto paramsExamples() -> std::vector<Params> {
    return std::vector<Params>({
        SpiralSimilarity3Impl::exp(
            {Scalar(0.2), Scalar(0.5), Scalar(0.0), Scalar(1.0)}),
        SpiralSimilarity3Impl::exp(
            {Scalar(0.2), Scalar(0.5), Scalar(-1.0), Scalar(1.1)}),
        SpiralSimilarity3Impl::exp(
            {Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(1.1)}),
        SpiralSimilarity3Impl::exp(
            {Scalar(0.0), Scalar(0.0), Scalar(0.00001), Scalar(0)}),
        SpiralSimilarity3Impl::exp(
            {Scalar(0.0), Scalar(0.0), Scalar(0.00001), Scalar(0.00001)}),
        SpiralSimilarity3Impl::exp(
            {Scalar(0.5 * kPi<Scalar>), Scalar(0.9), Scalar(0.0), Scalar(0.0)}),
        SpiralSimilarity3Impl::exp(
            {Scalar(0.5 * kPi<Scalar> + 0.00001),
             Scalar(0.0),
             Scalar(0.0),
             Scalar(0.9)}),
    });
  }

  static auto invalidParamsExamples() -> std::vector<Params> {
    return std::vector<Params>({
        Params::Zero(),
        -Params::Ones(),
        -Params::UnitX(),
    });
  }
};

}  // namespace lie
}  // namespace sophus
