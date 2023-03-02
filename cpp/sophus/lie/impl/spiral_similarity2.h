// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once
#include "sophus/concepts/lie_group.h"
#include "sophus/lie/impl/rotation2.h"
#include "sophus/lie/impl/sim_mat_w.h"
#include "sophus/linalg/complex.h"
#include "sophus/linalg/unit_vector.h"

namespace sophus {
namespace lie {

template <class TScalar, int kDim = 2>
class SpiralSimilarity2Impl {
 public:
  static_assert(kDim == 2);

  using Scalar = TScalar;
  using Complex = ComplexImpl<TScalar>;

  static bool constexpr kIsOriginPreserving = true;
  static bool constexpr kIsAxisDirectionPreserving = false;
  static bool constexpr kIsDirectionVectorPreserving = false;
  static bool constexpr kIsShapePreserving = false;
  static bool constexpr kIisSizePreserving = true;
  static bool constexpr kIisParallelLinePreserving = true;

  static int const kDof = 2;
  static int const kNumParams = 2;
  static int const kPointDim = 2;
  static int const kAmbientDim = 2;

  // constructors and factories

  static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
    return Eigen::Vector<Scalar, 2>(1.0, 0.0);
  }

  static auto areParamsValid(
      Eigen::Vector<Scalar, kNumParams> const& non_zero_complex)
      -> sophus::Expected<Success> {
    static const Scalar kThr = kEpsilon<Scalar> * kEpsilon<Scalar>;
    const Scalar squared_norm = non_zero_complex.squaredNorm();
    using std::abs;
    if (!(squared_norm > kThr || squared_norm < 1.0 / kThr)) {
      return SOPHUS_UNEXPECTED(
          "complex number ({}, {}) is too large or too small.\n"
          "Squared norm: {}, thr: {}",
          non_zero_complex[0],
          non_zero_complex[1],
          squared_norm,
          kThr);
    }
    return sophus::Expected<Success>{};
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> const& angle_logscale)
      -> Eigen::Vector<Scalar, kNumParams> {
    using std::exp;
    using std::max;
    using std::min;

    Scalar const sigma = angle_logscale[1];
    Scalar s = exp(sigma);
    // Ensuring proper scale
    s = max(s, kEpsilonPlus<Scalar>);
    s = min(s, Scalar(1.) / kEpsilonPlus<Scalar>);
    Eigen::Vector2<Scalar> z =
        Rotation2Impl<Scalar>::exp(angle_logscale.template head<1>());
    z *= s;
    return z;
  }

  static auto log(Eigen::Vector<Scalar, kNumParams> const& complex)
      -> Eigen::Vector<Scalar, kDof> {
    using std::log;
    Eigen::Vector<Scalar, kDof> theta_sigma;
    theta_sigma[0] = Rotation2Impl<Scalar>::log(complex)[0];
    theta_sigma[1] = log(complex.norm());
    return theta_sigma;
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& angle_logscale)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return Eigen::Matrix<Scalar, 2, 2>{
        {angle_logscale[1], -angle_logscale[0]},
        {angle_logscale[0], angle_logscale[1]}};
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return Eigen::Matrix<Scalar, kDof, 1>{mat(1, 0), mat(0, 0)};
  }

  static auto adj(Eigen::Vector<Scalar, kNumParams> const& /*unused*/)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, 2, 2>::Identity();
  }

  // group operations

  static auto inverse(Eigen::Vector<Scalar, kNumParams> const& non_zero_complex)
      -> Eigen::Vector<Scalar, kNumParams> {
    Scalar squared_scale = non_zero_complex.squaredNorm();
    return Eigen::Vector<Scalar, kNumParams>(
        non_zero_complex.x() / squared_scale,
        -non_zero_complex.y() / squared_scale);
  }

  static auto multiplication(
      Eigen::Vector<Scalar, kNumParams> const& lhs_params,
      Eigen::Vector<Scalar, kNumParams> const& rhs_params)
      -> Eigen::Vector<Scalar, kNumParams> {
    auto result = Complex::multiplication(lhs_params, rhs_params);
    Scalar const squared_scale = result.squaredNorm();

    if (squared_scale < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      /// Saturation to ensure class invariant.
      result.normalize();
      result *= kEpsilonPlus<Scalar>;
    }
    if (squared_scale > Scalar(1.) / (kEpsilon<Scalar> * kEpsilon<Scalar>)) {
      /// Saturation to ensure class invariant.
      result.normalize();
      result /= kEpsilonPlus<Scalar>;
    }
    return result;
  }

  // Point actions
  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& non_zero_complex,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kPointDim> {
    return matrix(non_zero_complex) * point;
  }

  static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }

  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& non_zero_quat,
      UnitVector<Scalar, kPointDim> const& direction_vector)
      -> UnitVector<Scalar, kPointDim> {
    return UnitVector<Scalar, kPointDim>::fromParams(
        Rotation2Impl<Scalar>::matrix(non_zero_quat.normalized()) *
        direction_vector.vector());
  }

  // matrices

  static auto compactMatrix(
      Eigen::Vector<Scalar, kNumParams> const& non_zero_complex)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, 2, 2>{
        {non_zero_complex.x(), -non_zero_complex.y()},
        {non_zero_complex.y(), non_zero_complex.x()}};
  }

  static auto matrix(Eigen::Vector<Scalar, kNumParams> const& non_zero_complex)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return compactMatrix(non_zero_complex);
  }

  // Sub-group concepts
  static auto matV(
      Eigen::Vector<Scalar, kNumParams> const& non_zero_complex,
      Eigen::Vector<Scalar, kDof> const& angle_logscale)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return details::calcW<Scalar, 2>(
        Rotation2Impl<Scalar>::hat(angle_logscale.template head<1>()),
        angle_logscale[0],
        angle_logscale[1]);
  }

  static auto matVInverse(
      Eigen::Vector<Scalar, kNumParams> const& non_zero_complex,
      Eigen::Vector<Scalar, kDof> const& angle_logscale)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return details::calcWInv<Scalar, 2>(
        Rotation2Impl<Scalar>::hat(angle_logscale.template head<1>()),
        angle_logscale[0],
        angle_logscale[1],
        non_zero_complex.norm());
  }

  static auto topRightAdj(
      Eigen::Vector<Scalar, kNumParams> const& /*unused*/,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Matrix<Scalar, 2, 2>{
        {point[1], -point[0]}, {-point[0], -point[1]}};
  }

  // derivatives

  static auto dxExpX(Eigen::Vector<Scalar, kDof> const& a)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    using std::cos;
    using std::exp;
    using std::sin;
    Scalar const theta = a[0];
    Scalar const sigma = a[1];

    Eigen::Matrix<Scalar, 2, 2> d;
    // clang-format off
    d << -sin(theta), cos(theta),
          cos(theta), sin(theta);
    // clang-format on
    return d * exp(sigma);
  }

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, 2, 2> d;
    Scalar const i(1.);
    Scalar const o(0.);
    // clang-format off
    d << o, i,
         i, o;
    // clang-format on
    return d;
  }

  static auto dxExpXTimesPointAt0(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, 2, 2> d;
    d << Rotation2Impl<Scalar>::dxExpXTimesPointAt0(point), point;
    return d;
  }

  static auto dxThisMulExpXAt0(
      Eigen::Vector<Scalar, kNumParams> const& non_zero_complex)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, 2, 2> d;
    // clang-format off
    d << -non_zero_complex.y(), non_zero_complex.x(),
          non_zero_complex.x(), non_zero_complex.y();
    // clang-format on
    return d;
  }

  static auto dxLogThisInvTimesXAtThis(
      Eigen::Vector<Scalar, kNumParams> const& non_zero_complex)
      -> Eigen::Matrix<Scalar, kDof, kNumParams> {
    Eigen::Matrix<Scalar, 2, 2> d;
    const Scalar norm_sq_inv = Scalar(1.) / non_zero_complex.squaredNorm();
    // clang-format off
    d << -non_zero_complex.y(), non_zero_complex.x(),
          non_zero_complex.x(), non_zero_complex.y();
    // clang-format on
    return d * norm_sq_inv;
  }

  // for tests

  static auto tangentExamples() -> std::vector<Eigen::Vector<Scalar, kDof>> {
    return std::vector<Eigen::Vector<Scalar, kDof>>({
        Eigen::Vector<Scalar, kDof>{0.2, 1},
        Eigen::Vector<Scalar, kDof>{0.2, 1.1},
        Eigen::Vector<Scalar, kDof>{0.0, 1.1},
        Eigen::Vector<Scalar, kDof>{0.00001, 0},
        Eigen::Vector<Scalar, kDof>{0.00001, 0.00001},
        Eigen::Vector<Scalar, kDof>{0.5 * kPi<Scalar>, 0.9},
        Eigen::Vector<Scalar, kDof>{0.5 * kPi<Scalar> + 0.00001, 0.2},
    });
  }

  static auto paramsExamples()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    return std::vector<Eigen::Vector<Scalar, kNumParams>>({
        SpiralSimilarity2Impl::exp({0.2, 1}),
        SpiralSimilarity2Impl::exp({0.2, 1.1}),
        SpiralSimilarity2Impl::exp({0.0, 1.1}),
        SpiralSimilarity2Impl::exp({0.00001, 0}),
        SpiralSimilarity2Impl::exp({0.00001, 0.00001}),
        SpiralSimilarity2Impl::exp({0.5 * kPi<Scalar>, 0.9}),
        SpiralSimilarity2Impl::exp({0.5 * kPi<Scalar> + 0.00001, 0.2}),
    });
  }

  static auto invalidParamsExamples()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    return std::vector<Eigen::Vector<Scalar, kNumParams>>({
        Eigen::Vector<Scalar, kNumParams>::Zero(),
        -Eigen::Vector<Scalar, kNumParams>::Ones(),
        -Eigen::Vector<Scalar, kNumParams>::UnitX(),
    });
  }
};

}  // namespace lie
}  // namespace sophus
