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
class IdentityImpl {
 public:
  using Scalar = TScalar;

  static bool constexpr kIsOriginPreserving = true;
  static bool constexpr kIsAxisDirectionPreserving = true;
  static bool constexpr kIsDirectionVectorPreserving = true;
  static bool constexpr kIsShapePreserving = true;
  static bool constexpr kIisSizePreserving = true;
  static bool constexpr kIisParallelLinePreserving = true;

  static int const kDof = 0;
  static int const kNumParams = 0;
  static int const kPointDim = kDim;
  static int const kAmbientDim = kDim;

  // constructors and factories

  static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
    return Eigen::Vector<Scalar, kNumParams>::Zero();
  }

  static auto areParamsValid(
      Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> sophus::Expected<Success> {
    return sophus::Expected<Success>{};
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Vector<Scalar, kNumParams> {
    return tangent;
  }

  static auto log(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kDof> {
    return params;
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> mat;
    mat.setZero();
    return mat;
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return Eigen::Matrix<Scalar, kDof, 1>();
  }

  static auto adj(Eigen::Vector<Scalar, kNumParams> const& /*unused*/)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, kDof, kDof>::Identity();
  }

  // group operations

  static auto inverse(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kNumParams> {
    return params;
  }

  static auto multiplication(
      Eigen::Vector<Scalar, kNumParams> const& lhs_params,
      Eigen::Vector<Scalar, kNumParams> const& rhs_params)
      -> Eigen::Vector<Scalar, kNumParams> {
    return lhs_params;
  }

  // Point actions
  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& params,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kPointDim> {
    return point;
  }

  static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }

  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& params,
      UnitVector<Scalar, kPointDim> const& point)
      -> UnitVector<Scalar, kPointDim> {
    return point;
  }

  // Matrices

  static auto compactMatrix(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kPointDim, kAmbientDim> {
    return Eigen::Matrix<Scalar, kPointDim, kAmbientDim>::Identity();
  }

  static auto matrix(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return compactMatrix(params);
  }

  // subgroup concepts

  static auto matV(
      Eigen::Vector<Scalar, kNumParams> const& /*unused*/,
      Eigen::Vector<Scalar, kDof> const& /*unused*/)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
  }

  static auto matVInverse(
      Eigen::Vector<Scalar, kNumParams> const& /*unused*/,
      Eigen::Vector<Scalar, kDof> const& /*unused*/)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
  }

  static auto topRightAdj(
      Eigen::Vector<Scalar, kNumParams> const& params,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Matrix<Scalar, kPointDim, kDof>::Zero();
  }

  // derivatives
  static auto dxExpX(Eigen::Vector<Scalar, kDof> const& /*unused*/)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, kNumParams, 0>::Identity();
  }

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, kNumParams, 0>::Identity();
  }

  static auto dxExpXTimesPointAt0(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Matrix<Scalar, kPointDim, 0>::Identity();
  }

  static auto dxThisMulExpXAt0(
      Eigen::Vector<Scalar, kNumParams> const& unit_complex)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, kNumParams, 0>::Zero();
  }

  static auto dxLogThisInvTimesXAtThis(
      Eigen::Vector<Scalar, kNumParams> const& unit_quat)
      -> Eigen::Matrix<Scalar, kDof, kNumParams> {
    return Eigen::Matrix<Scalar, kDof, kNumParams>::Identity();
  }

  // for tests

  static auto tangentExamples() -> std::vector<Eigen::Vector<Scalar, kDof>> {
    return std::vector<Eigen::Vector<Scalar, kDof>>();
  }

  static auto paramsExamples()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    return std::vector<Eigen::Vector<Scalar, kNumParams>>();
  }

  static auto invalidParamsExamples()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    return std::vector<Eigen::Vector<Scalar, kNumParams>>();
  }
};

}  // namespace lie
}  // namespace sophus
