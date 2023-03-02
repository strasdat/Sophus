// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once
#include "sophus/concepts/lie_group.h"
#include "sophus/linalg/unit_vector.h"

namespace sophus {
namespace lie {

template <class TScalar, int kDim>
class ScalingImpl {
 public:
  using Scalar = TScalar;
  static int const kDof = kDim;
  static int const kNumParams = kDim;
  static int const kPointDim = kDim;
  static int const kAmbientDim = kDim;

  static bool constexpr kIsOriginPreserving = true;
  static bool constexpr kIsAxisDirectionPreserving = false;
  static bool constexpr kIsDirectionVectorPreserving = false;
  static bool constexpr kIsShapePreserving = true;
  static bool constexpr kIisSizePreserving = true;
  static bool constexpr kIisParallelLinePreserving = true;

  // constructors and factories

  static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
    return Eigen::Vector<Scalar, kDim>::Ones();
  }

  static auto areParamsValid(
      Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> sophus::Expected<Success> {
    static const Scalar kThr = kEpsilon<Scalar>;

    if (!(scale_factors.array() >= kThr).all()) {
      return SOPHUS_UNEXPECTED(
          "scale factors ({}) not positive.\n",
          "thr: {}",
          scale_factors.transpose(),
          kThr);
    }
    return sophus::Expected<Success>{};
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> const& log_scale_factors)
      -> Eigen::Vector<Scalar, kNumParams> {
    using std::exp;
    return log_scale_factors.array().exp();
  }

  static auto log(Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> Eigen::Vector<Scalar, kDof> {
    using std::log;
    return scale_factors.array().log();
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& scale_factors)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> mat;
    mat.setZero();
    for (int i = 0; i < kDof; ++i) {
      mat.diagonal()[i] = scale_factors[i];
    }
    return mat;
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return mat.diagonal();
  }

  static auto adj(Eigen::Vector<Scalar, kNumParams> const& /*unused*/)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, kDof, kDof>::Identity();
  }

  // group operations

  static auto inverse(Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, kDim> params;
    for (int i = 0; i < kDof; ++i) {
      params[i] = 1.0 / scale_factors[i];
    }
    return params;
  }

  static auto multiplication(
      Eigen::Vector<Scalar, kNumParams> const& lhs_params,
      Eigen::Vector<Scalar, kNumParams> const& rhs_params)
      -> Eigen::Vector<Scalar, kNumParams> {
    return lhs_params.array() * rhs_params.array();
  }

  // Point actions

  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& scale_factors,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kPointDim> {
    return scale_factors.array() * point.array();
  }

  static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }

  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& scale_factors,
      UnitVector<Scalar, kPointDim> const& direction_vector)
      -> UnitVector<Scalar, kPointDim> {
    return UnitVector<Scalar, kPointDim>::fromVectorAndNormalize(
        action(scale_factors, direction_vector.vector()));
  }
  // Matrices

  static auto compactMatrix(
      Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> Eigen::Matrix<Scalar, kPointDim, kAmbientDim> {
    return hat(scale_factors);
  }

  static auto matrix(Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return compactMatrix(scale_factors);
  }

  // subgroup concepts

  static auto matV(
      Eigen::Vector<Scalar, kNumParams> const& params,
      Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    Eigen::Matrix<Scalar, kPointDim, kPointDim> mat =
        Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
    for (int i = 0; i < kDof; ++i) {
      mat(i, i) = (params[i] - 1.0) / tangent[i];
    }
    return mat;
  }

  static auto matVInverse(
      Eigen::Vector<Scalar, kNumParams> const& params,
      Eigen::Vector<Scalar, kNumParams> const& tangent)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    Eigen::Matrix<Scalar, kPointDim, kPointDim> mat =
        Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
    for (int i = 0; i < kDof; ++i) {
      mat(i, i) = tangent[i] / (params[i] - 1.0);
    }
    return mat;
  }

  static auto topRightAdj(
      Eigen::Vector<Scalar, kNumParams> const& params,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return matrix(-point);
  }

  // derivatives
  static auto dxExpX(Eigen::Vector<Scalar, kDof> const& /*unused*/)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, kNumParams, kDof>::Identity();
  }

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, kNumParams, kDof>::Identity();
  }

  static auto dxExpXTimesPointAt0(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, kPointDim, kDof> j;
    j.setZero();
    j.diagonal() = point;
    return j;
  }

  static auto dxThisMulExpXAt0(
      Eigen::Vector<Scalar, kNumParams> const& unit_quat)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, kNumParams, kDof> j;
    j.setZero();
    j.diagonal() = unit_quat;
    return j;
  }

  static auto dxLogThisInvTimesXAtThis(
      Eigen::Vector<Scalar, kNumParams> const& unit_quat)
      -> Eigen::Matrix<Scalar, kDof, kNumParams> {
    Eigen::Matrix<Scalar, kDof, kNumParams> j;
    j.setZero();
    j.diagonal() = 1.0 / unit_quat.array();
    return j;
  }

  // for tests

  static auto tangentExamples() -> std::vector<Eigen::Vector<Scalar, kDof>> {
    if constexpr (kPointDim == 2) {
      return std::vector<Eigen::Vector<Scalar, kDof>>({
          Eigen::Vector<Scalar, kDof>({std::exp(1.0), std::exp(1.0)}),
          Eigen::Vector<Scalar, kDof>({1.1, 1.1}),
          Eigen::Vector<Scalar, kDof>({2.0, 1.1}),
          Eigen::Vector<Scalar, kDof>({2.0, std::exp(1.0)}),
      });
    } else {
      if constexpr (kPointDim == 3) {
        return std::vector<Eigen::Vector<Scalar, kDof>>({
            Eigen::Vector<Scalar, kDof>(
                {std::exp(1.0), std::exp(1.0), std::exp(1.0)}),
            Eigen::Vector<Scalar, kDof>({1.1, 1.1, 1.7}),
            Eigen::Vector<Scalar, kDof>({2.0, 1.1, 2.0}),
            Eigen::Vector<Scalar, kDof>({2.0, std::exp(1.0), 2.2}),
        });
      }
    }
  }

  static auto paramsExamples()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    if constexpr (kPointDim == 2) {
      return std::vector<Eigen::Vector<Scalar, kNumParams>>(
          {Eigen::Vector<Scalar, kNumParams>({1.0, 1.0}),
           Eigen::Vector<Scalar, kNumParams>({1.0, 2.0}),
           Eigen::Vector<Scalar, kNumParams>({1.5, 1.0}),
           Eigen::Vector<Scalar, kNumParams>({5.0, 1.237})});
    } else {
      if constexpr (kPointDim == 3) {
        return std::vector<Eigen::Vector<Scalar, kNumParams>>(
            {Eigen::Vector<Scalar, kNumParams>({1.0, 1.0, 1.0}),
             Eigen::Vector<Scalar, kNumParams>({1.0, 2.0, 1.05}),
             Eigen::Vector<Scalar, kNumParams>({1.5, 1.0, 2.8}),
             Eigen::Vector<Scalar, kNumParams>({5.0, 1.237, 2})});
      }
    }
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

template <class TScalar>
using Scaling2Impl = ScalingImpl<TScalar, 2>;

template <class TScalar>
using Scaling3Impl = ScalingImpl<TScalar, 3>;

}  // namespace lie
}  // namespace sophus
