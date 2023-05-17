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
#include "sophus/manifold/complex.h"
#include "sophus/manifold/unit_vector.h"

namespace sophus {
namespace lie {

template <class TScalar>
class SpiralSimilarity2Impl {
 public:
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
    return Eigen::Vector<Scalar, 2>(1.0, 0.0);
  }

  static auto areParamsValid(Params const& non_zero_complex)
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

  static auto hasShortestPathAmbiguity(Params const& non_zero_complex) -> bool {
    return Rotation2Impl<Scalar>::hasShortestPathAmbiguity(
        non_zero_complex.normalized());
  }

  // Manifold / Lie Group concepts

  static auto exp(Tangent const& angle_logscale) -> Params {
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

  static auto log(Params const& complex) -> Tangent {
    using std::log;
    Tangent theta_sigma;
    theta_sigma[0] =
        Eigen::Vector<Scalar, 1>{atan2(complex.y(), complex.x())}[0];
    theta_sigma[1] = log(complex.norm());
    return theta_sigma;
  }

  static auto hat(Tangent const& angle_logscale)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return Eigen::Matrix<Scalar, 2, 2>{
        {angle_logscale[1], -angle_logscale[0]},
        {angle_logscale[0], angle_logscale[1]}};
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return Eigen::Matrix<Scalar, kDof, 1>{mat(1, 0), mat(0, 0)};
  }

  // group operations

  static auto inverse(Params const& non_zero_complex) -> Params {
    Scalar squared_scale = non_zero_complex.squaredNorm();
    return Params(
        non_zero_complex.x() / squared_scale,
        -non_zero_complex.y() / squared_scale);
  }

  template <class TCompatibleScalar>
  static auto multiplication(
      Params const& lhs_params,
      Eigen::Vector<TCompatibleScalar, kNumParams> const& rhs_params)
      -> ParamsReturn<TCompatibleScalar> {
    auto result = Complex::multiplication(lhs_params, rhs_params);
    using ScalarReturn = typename ParamsReturn<TCompatibleScalar>::Scalar;
    ScalarReturn const squared_scale = result.squaredNorm();

    if (squared_scale < kEpsilon<Scalar> * kEpsilon<Scalar>) {
      /// Saturation to ensure class invariant.
      result.normalize();
      result *= kEpsilonPlus<ScalarReturn>;
    }
    if (squared_scale >
        Scalar(1.) / (kEpsilon<ScalarReturn> * kEpsilon<ScalarReturn>)) {
      /// Saturation to ensure class invariant.
      result.normalize();
      result /= kEpsilonPlus<ScalarReturn>;
    }
    return result;
  }

  // Group actions
  template <class TCompatibleScalar>
  static auto action(
      Params const& non_zero_complex,
      Eigen::Vector<TCompatibleScalar, kPointDim> const& point)
      -> PointReturn<TCompatibleScalar> {
    return matrix(non_zero_complex) * point;
  }

  static auto toAmbient(Point const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }

  template <class TCompatibleScalar>
  static auto action(
      Params const& non_zero_complex,
      UnitVector<TCompatibleScalar, kPointDim> const& direction_vector)
      -> UnitVectorReturn<TCompatibleScalar> {
    return UnitVectorReturn<TCompatibleScalar>::fromParams(
        Rotation2Impl<Scalar>::matrix(non_zero_complex.normalized()) *
        direction_vector.params());
  }

  static auto adj(Params const& /*unused*/)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, 2, 2>::Identity();
  }

  // matrices

  static auto compactMatrix(Params const& non_zero_complex)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, 2, 2>{
        {Scalar(non_zero_complex.x()), Scalar(-non_zero_complex.y())},
        {Scalar(non_zero_complex.y()), Scalar(non_zero_complex.x())}};
  }

  static auto matrix(Params const& non_zero_complex)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return compactMatrix(non_zero_complex);
  }

  // Sub-group concepts
  static auto matV(Params const&, Tangent const& angle_logscale)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return details::calcW<Scalar, 2>(
        Rotation2Impl<Scalar>::hat(angle_logscale.template head<1>()),
        angle_logscale[0],
        angle_logscale[1]);
  }

  static auto matVInverse(
      Params const& non_zero_complex, Tangent const& angle_logscale)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return details::calcWInv<Scalar, 2>(
        Rotation2Impl<Scalar>::hat(angle_logscale.template head<1>()),
        angle_logscale[0],
        angle_logscale[1],
        non_zero_complex.norm());
  }

  static auto adjOfTranslation(Params const& /*unused*/, Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Matrix<Scalar, 2, 2>{
        {point[1], -point[0]}, {-point[0], -point[1]}};
  }

  static auto adOfTranslation(Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, 2, 2> mat;
    mat.col(0) = Eigen::Vector2<Scalar>(point[1], -point[0]);
    mat.col(1) = -point;
    return mat;
  }

  // derivatives

  static auto ad(Tangent const& /*unused*/)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, 2, 2>::Zero();
  }

  static auto dxExpX(Tangent const& a)
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

  static auto dxExpXTimesPointAt0(Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, 2, 2> d;
    d << Rotation2Impl<Scalar>::dxExpXTimesPointAt0(point), point;
    return d;
  }

  static auto dxThisMulExpXAt0(Params const& non_zero_complex)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, 2, 2> d;
    // clang-format off
    d << -non_zero_complex.y(), non_zero_complex.x(),
          non_zero_complex.x(), non_zero_complex.y();
    // clang-format on
    return d;
  }

  static auto dxLogThisInvTimesXAtThis(Params const& non_zero_complex)
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

  static auto tangentExamples() -> std::vector<Tangent> {
    return std::vector<Tangent>({
        Tangent{0.2, 1},
        Tangent{0.2, 1.1},
        Tangent{0.0, 1.1},
        Tangent{0.00001, 0},
        Tangent{0.00001, 0.00001},
        Tangent{0.5 * kPi<Scalar>, 0.9},
        Tangent{0.5 * kPi<Scalar> + 0.00001, 0.2},
    });
  }

  static auto paramsExamples() -> std::vector<Params> {
    return std::vector<Params>({
        SpiralSimilarity2Impl::exp({0.2, 1}),
        SpiralSimilarity2Impl::exp({0.2, 1.1}),
        SpiralSimilarity2Impl::exp({0.0, 1.1}),
        SpiralSimilarity2Impl::exp({0.00001, 0}),
        SpiralSimilarity2Impl::exp({0.00001, 0.00001}),
        SpiralSimilarity2Impl::exp({0.5 * kPi<Scalar>, 0.9}),
        SpiralSimilarity2Impl::exp({0.5 * kPi<Scalar> + 0.00001, 0.2}),
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
