// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/lie_group.h"
#include "sophus/manifold/complex.h"
#include "sophus/manifold/unit_vector.h"

namespace sophus {
namespace lie {

template <class TScalar>
class Rotation2Impl {
 public:
  using Scalar = TScalar;
  using Complex = ComplexImpl<TScalar>;

  static bool constexpr kIsOriginPreserving = true;
  static bool constexpr kIsAxisDirectionPreserving = false;
  static bool constexpr kIsDirectionVectorPreserving = false;
  static bool constexpr kIsShapePreserving = true;
  static bool constexpr kIisSizePreserving = true;
  static bool constexpr kIisParallelLinePreserving = true;

  static int const kDof = 1;
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

  static auto areParamsValid(Params const& unit_complex)
      -> sophus::Expected<Success> {
    static const Scalar kThr = kEpsilon<Scalar>;
    const Scalar squared_norm = unit_complex.squaredNorm();
    using std::abs;
    if (!(abs(squared_norm - 1.0) <= kThr)) {
      return SOPHUS_UNEXPECTED(
          "complex number ({}, {}) is not unit length.\n"
          "Squared norm: {}, thr: {}",
          unit_complex[0],
          unit_complex[1],
          squared_norm,
          kThr);
    }
    return sophus::Expected<Success>{};
  }

  static auto hasShortestPathAmbiguity(Params const& foo_from_bar) -> bool {
    using std::abs;
    TScalar angle = abs(Rotation2Impl::log(foo_from_bar)[0]);
    TScalar const k_pi = kPi<TScalar>;  // NOLINT
    return abs(angle - k_pi) / (angle + k_pi) < kEpsilon<TScalar>;
  }

  // Manifold / Lie Group concepts

  static auto exp(Tangent const& angle) -> Params {
    using std::cos;
    using std::sin;
    return Eigen::Vector<Scalar, 2>(cos(angle[0]), sin(angle[0]));
  }

  static auto log(Params const& unit_complex) -> Tangent {
    using std::atan2;
    return Eigen::Vector<Scalar, 1>{atan2(unit_complex.y(), unit_complex.x())};
  }

  static auto hat(Tangent const& angle)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return Eigen::Matrix<Scalar, 2, 2>{
        {Scalar(0.0), Scalar(-angle[0])}, {Scalar(angle[0]), Scalar(0.0)}};
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return Eigen::Matrix<Scalar, kDof, 1>{mat(1, 0)};
  }

  // group operations

  static auto inverse(Params const& unit_complex) -> Params {
    return Params(unit_complex.x(), -unit_complex.y());
  }

  template <class TCompatibleScalar>
  static auto multiplication(
      Params const& lhs_params,
      Eigen::Matrix<TCompatibleScalar, 2, 1> const& rhs_params)
      -> ParamsReturn<TCompatibleScalar> {
    auto result = Complex::multiplication(lhs_params, rhs_params);
    auto const squared_norm = result.squaredNorm();

    // We can assume that the squared-norm is close to 1 since we deal with a
    // unit complex number. Due to numerical precision issues, there might
    // be a small drift after pose concatenation. Hence, we need to renormalizes
    // the complex number here.
    // Since squared-norm is close to 1, we do not need to calculate the costly
    // square-root, but can use an approximation around 1 (see
    // http://stackoverflow.com/a/12934750 for details).
    if (squared_norm != 1.0) {
      auto const scale = 2.0 / (1.0 + squared_norm);
      return scale * result;
    }
    return result;
  }

  // Group actions
  template <class TCompatibleScalar>
  static auto action(
      Params const& unit_complex,
      Eigen::Matrix<TCompatibleScalar, 2, 1> const& point)
      -> PointReturn<TCompatibleScalar> {
    return Complex::multiplication(unit_complex, point);
  }

  template <class TCompatibleScalar>
  static auto action(
      Params const& unit_complex,
      UnitVector<TCompatibleScalar, kPointDim> const& direction_vector)
      -> UnitVectorReturn<TCompatibleScalar> {
    return UnitVectorReturn<TCompatibleScalar>::fromParams(
        Complex::multiplication(unit_complex, direction_vector.params()));
  }

  static auto toAmbient(Point const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }

  static auto adj(Params const& /*unused*/)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, 1, 1>::Identity();
  }

  // matrices

  static auto compactMatrix(Params const& unit_complex)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, 2, 2>{
        {unit_complex.x(), -unit_complex.y()},
        {unit_complex.y(), unit_complex.x()}};
  }

  static auto matrix(Params const& unit_complex)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return compactMatrix(unit_complex);
  }

  // Sub-group concepts
  static auto matV(Params const& params, Tangent const& theta)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    Scalar sin_theta_by_theta;
    Scalar one_minus_cos_theta_by_theta;
    using std::abs;

    if (abs(theta[0]) < kEpsilon<Scalar>) {
      Scalar theta_sq = theta[0] * theta[0];
      sin_theta_by_theta = Scalar(1.) - Scalar(1. / 6.) * theta_sq;
      one_minus_cos_theta_by_theta =
          Scalar(0.5) * theta[0] - Scalar(1. / 24.) * theta[0] * theta_sq;
    } else {
      sin_theta_by_theta = params.y() / theta[0];
      one_minus_cos_theta_by_theta = (Scalar(1.) - params.x()) / theta[0];
    }
    return Eigen::Matrix<Scalar, 2, 2>(
        {{sin_theta_by_theta, -one_minus_cos_theta_by_theta},
         {one_minus_cos_theta_by_theta, sin_theta_by_theta}});
  }

  static auto matVInverse(Params const& z, Tangent const& theta)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    Scalar halftheta = Scalar(0.5) * theta[0];
    Scalar halftheta_by_tan_of_halftheta;

    Scalar real_minus_one = z.x() - Scalar(1.);
    if (abs(real_minus_one) < kEpsilon<Scalar>) {
      halftheta_by_tan_of_halftheta =
          Scalar(1.) - Scalar(1. / 12) * theta[0] * theta[0];
    } else {
      halftheta_by_tan_of_halftheta = -(halftheta * z.y()) / (real_minus_one);
    }
    Eigen::Matrix<Scalar, 2, 2> v_inv;
    v_inv << halftheta_by_tan_of_halftheta, halftheta, -halftheta,
        halftheta_by_tan_of_halftheta;
    return v_inv;
  }

  static auto adjOfTranslation(Params const& /*unused*/, Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Matrix<Scalar, 2, 1>(point[1], -point[0]);
  }

  static auto adOfTranslation(Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Vector2<Scalar>(point[1], -point[0]);
  }

  // derivatives
  static auto ad(Tangent const&) -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, 1, 1>::Zero();
  }

  static auto dxExpX(Tangent const& theta)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    using std::cos;
    using std::sin;
    return Eigen::Matrix<Scalar, kNumParams, kDof>(
        -sin(theta[0]), cos(theta[0]));
  }

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, kNumParams, kDof>(0.0, 1.0);
  }

  static auto dxExpXTimesPointAt0(Point const& point)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, 2, 1>(-point[1], point[0]);
  }

  static auto dxThisMulExpXAt0(Params const& unit_complex)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return -Eigen::Matrix<Scalar, 2, 1>(unit_complex[1], -unit_complex[0]);
  }

  static auto dxLogThisInvTimesXAtThis(Params const& unit_complex)
      -> Eigen::Matrix<Scalar, kDof, kNumParams> {
    return Eigen::Matrix<Scalar, 1, 2>(-unit_complex[1], unit_complex[0]);
  }

  // for tests

  static auto tangentExamples() -> std::vector<Tangent> {
    return std::vector<Tangent>({
        Tangent{Scalar(0.0)},
        Tangent{Scalar(0.00001)},
        Tangent{Scalar(1.0)},
        Tangent{Scalar(-1.0)},
        Tangent{Scalar(5.0)},
        Tangent{Scalar(0.5 * kPi<Scalar>)},
        Tangent{Scalar(0.5 * kPi<Scalar> + 0.00001)},
    });
  }

  static auto paramsExamples() -> std::vector<Params> {
    return std::vector<Params>({
        Rotation2Impl::exp(Tangent{Scalar(0.0)}),
        Rotation2Impl::exp(Tangent{Scalar(1.0)}),
        Rotation2Impl::exp(Tangent{Scalar(0.5 * kPi<Scalar>)}),
        Rotation2Impl::exp(Tangent{Scalar(kPi<Scalar>)}),
    });
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
