// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/lie_group.h"
#include "sophus/manifold/unit_vector.h"

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

  using Tangent = Eigen::Vector<Scalar, kDof>;
  using Params = Eigen::Vector<Scalar, kNumParams>;
  using Point = Eigen::Vector<Scalar, kPointDim>;

  // constructors and factories

  static auto identityParams() -> Params { return Params::Zero(); }

  static auto areParamsValid(Params const& scale_factors)
      -> sophus::Expected<Success> {
    return sophus::Expected<Success>{};
  }

  static auto adj(Params const& /*unused*/)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, kDof, kDof>::Identity();
  }

  // Manifold / Lie Group concepts

  static auto exp(Tangent const& tangent) -> Params { return tangent; }

  static auto log(Params const& params) -> Tangent { return params; }

  static auto hat(Tangent const& tangent)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> mat;
    mat.setZero();
    return mat;
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return Eigen::Matrix<Scalar, kDof, 1>();
  }

  // group operations

  static auto inverse(Params const& params) -> Params { return params; }

  static auto multiplication(Params const& lhs_params, Params const& rhs_params)
      -> Params {
    return lhs_params;
  }

  // Point actions
  static auto action(Params const& params, Point const& point) -> Point {
    return point;
  }

  static auto toAmbient(Point const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }

  static auto action(
      Params const& params, UnitVector<Scalar, kPointDim> const& point)
      -> UnitVector<Scalar, kPointDim> {
    return point;
  }

  // Matrices

  static auto compactMatrix(Params const& params)
      -> Eigen::Matrix<Scalar, kPointDim, kAmbientDim> {
    return Eigen::Matrix<Scalar, kPointDim, kAmbientDim>::Identity();
  }

  static auto matrix(Params const& params)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return compactMatrix(params);
  }

  // subgroup concepts

  static auto matV(Params const& /*unused*/, Tangent const& /*unused*/)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
  }

  static auto matVInverse(Params const& /*unused*/, Tangent const& /*unused*/)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
  }

  static auto adjOfTranslation(Params const& params, Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Matrix<Scalar, kPointDim, kDof>::Zero();
  }

  static auto adOfTranslation(Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Matrix<Scalar, kPointDim, kDof>::Zero();
  }

  // derivatives
  static auto ad(Tangent const& /*unused*/)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, kDof, kDof>::Zero();
  }

  static auto dxExpX(Tangent const& /*unused*/)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, kNumParams, 0>::Identity();
  }

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, kNumParams, 0>::Identity();
  }

  static auto dxExpXTimesPointAt0(Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Matrix<Scalar, kPointDim, 0>::Identity();
  }

  static auto dxThisMulExpXAt0(Params const& unit_complex)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, kNumParams, 0>::Zero();
  }

  static auto dxLogThisInvTimesXAtThis(Params const& unit_quat)
      -> Eigen::Matrix<Scalar, kDof, kNumParams> {
    return Eigen::Matrix<Scalar, kDof, kNumParams>::Identity();
  }

  // for tests

  static auto tangentExamples() -> std::vector<Tangent> {
    return std::vector<Tangent>();
  }

  static auto paramsExamples() -> std::vector<Params> {
    return std::vector<Params>();
  }

  static auto invalidParamsExamples() -> std::vector<Params> {
    return std::vector<Params>();
  }
};

}  // namespace lie
}  // namespace sophus
