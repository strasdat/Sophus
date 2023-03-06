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
#include "sophus/lie/impl/scaling.h"
#include "sophus/linalg/unit_vector.h"

namespace sophus {
namespace lie {

template <class TScalar>
class RotoScaling3Impl {
 public:
  using Scalar = TScalar;
  static int const kDof = 6;
  static int const kNumParams = 7;
  static int const kPointDim = 3;
  static int const kAmbientDim = 3;

  using Tangent = Eigen::Vector<Scalar, kDof>;
  using Params = Eigen::Vector<Scalar, kNumParams>;
  using Point = Eigen::Vector<Scalar, kPointDim>;

  static bool constexpr kIsOriginPreserving = true;
  static bool constexpr kIsAxisDirectionPreserving = false;
  static bool constexpr kIsDirectionVectorPreserving = false;
  static bool constexpr kIsShapePreserving = false;
  static bool constexpr kIisSizePreserving = false;
  static bool constexpr kIisParallelLinePreserving = true;

  using Scaling = Scaling3Impl<Scalar>;
  using Rotation = Rotation3Impl<Scalar>;

  // constructors and factories

  static auto identityParams() -> Params {
    return Eigen::Vector<Scalar, 7>(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
  }

  static auto areParamsValid(Params const& params)
      -> sophus::Expected<Success> {
    FARM_TRY(
        auto roto_success, Rotation::areParamsValid(unitQuaternion(params)));
    FARM_TRY(
        auto scaling_success, Scaling::areParamsValid(scaleFactors(params)));

    return sophus::Expected<Success>{};
  }

  // Manifold / Lie Group concepts

  static auto exp(Tangent const& tangent) -> Params {
    return params(
        Scaling::exp(logFactors(tangent)),
        Rotation::exp(angleTimesAxis(tangent)));
  }

  static auto log(Params const& params) -> Tangent {
    return tangent(
        Rotation::log(unitQuaternion(params)),
        Scaling::log(scaleFactors(params)));
  }

  static auto hat(Tangent const& tangent)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return Rotation::hat(angleTimesAxis(tangent)) +
           Scaling::hat(logFactors(tangent));
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return tangent(Rotation::vee(mat), Scaling::vee(mat));
  }

  static auto adj(Params const& params) -> Eigen::Matrix<Scalar, kDof, kDof> {
    Eigen::Matrix<Scalar, 6, 6> mat_adjoint;
    mat_adjoint.setZero();

    mat_adjoint.template topLeftCorner<3, 3>() =
        Scaling::adj(scaleFactors(params));

    mat_adjoint.template bottomRightCorner<3, 3>() =
        Rotation::adj(unitQuaternion(params));

    return mat_adjoint;
  }

  // group operations

  static auto inverse(Params const& p) -> Params {
    Eigen::Vector3<Scalar> f = Scaling::inverse(scaleFactors(p));
    Eigen::Vector4<Scalar> q = Rotation::inverse(unitQuaternion(p));
    return params(f, q);
  }

  static auto multiplication(Params const& lhs_params, Params const& rhs_params)
      -> Params {
    Eigen::Matrix3<Scalar> d;
    d.setZero();
    d.diagonal() = scaleFactors(rhs_params);

    return params(
        Scaling::multiplication(
            scaleFactors(lhs_params),
            (Rotation::matrix(unitQuaternion(lhs_params)) * d).diagonal()),
        Rotation::multiplication(
            unitQuaternion(lhs_params), unitQuaternion(rhs_params)));
  }

  // Point actions

  static auto action(Params const& params, Point const& point) -> Point {
    return Rotation::action(
        unitQuaternion(params), Scaling::action(scaleFactors(params), point));
  }

  static auto toAmbient(Point const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }

  static auto action(
      Params const& scale_factors,
      UnitVector<Scalar, kPointDim> const& direction_vector)
      -> UnitVector<Scalar, kPointDim> {
    return UnitVector<Scalar, kPointDim>::fromVectorAndNormalize(
        action(scale_factors, direction_vector.vector()));
  }
  // Matrices

  static auto compactMatrix(Params const& params)
      -> Eigen::Matrix<Scalar, kPointDim, kAmbientDim> {
    return Rotation::compactMatrix(unitQuaternion(params)) *
           Scaling::compactMatrix(scaleFactors(params));
  }

  static auto matrix(Params const& params)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return compactMatrix(params);
  }

  // subgroup concepts

  static auto matV(Params const& params, Tangent const& tangent)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    using std::abs;
    Eigen::Matrix<Scalar, kPointDim, kPointDim> mat =
        Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
    for (int i = 0; i < kDof; ++i) {
      Scalar t = tangent[i];
      if (abs(t) < kEpsilon<Scalar>) {
        mat(i, i) = -1.0 + 2 * t - 1.5 * t * t;
      } else {
        mat(i, i) = (params[i] - 1.0) / tangent[i];
      }
    }
    return mat;
  }

  static auto matVInverse(Params const& params, Params const& tangent)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    Eigen::Matrix<Scalar, kPointDim, kPointDim> mat =
        Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
    for (int i = 0; i < kDof; ++i) {
      Scalar t = tangent[i];
      if (abs(t) < kEpsilon<Scalar>) {
        mat(i, i) = -1.0 - 2 * t - 2.5 * t * t;
      } else {
        mat(i, i) = tangent[i] / (params[i] - 1.0);
      }
    }
    return mat;
  }

  static auto topRightAdj(Params const& params, Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return matrix(-point);
  }

  // derivatives
  static auto dxExpX(Tangent const& /*unused*/)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, kNumParams, kDof>::Identity();
  }

  static auto dxExpXAt0() -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    return Eigen::Matrix<Scalar, kNumParams, kDof>::Identity();
  }

  static auto dxExpXTimesPointAt0(Point const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, kPointDim, kDof> j;
    j.setZero();
    j.diagonal() = point;
    return j;
  }

  static auto dxThisMulExpXAt0(Params const& params)
      -> Eigen::Matrix<Scalar, kNumParams, kDof> {
    Eigen::Matrix<Scalar, kNumParams, kDof> j;
    j.setZero();
    // j.diagonal() = unit_quat;
    return j;
  }

  static auto dxLogThisInvTimesXAtThis(Params const& unit_quat)
      -> Eigen::Matrix<Scalar, kDof, kNumParams> {
    Eigen::Matrix<Scalar, kDof, kNumParams> j;
    j.setZero();
    // j.diagonal() = 1.0 / unit_quat.array();
    return j;
  }

  // for tests

  static auto tangentExamples() -> std::vector<Tangent> {
    std::vector<Tangent> examples;
    for (auto const& angle_times_axis : Rotation::tangentExamples()) {
      for (auto const& log_factors : Scaling::tangentExamples()) {
        examples.push_back(tangent(angle_times_axis, log_factors));
      }
    }
    return examples;
  }

  static auto paramsExamples() -> std::vector<Params> {
    std::vector<Params> examples;
    for (auto const& scaling_factors : Scaling::paramsExamples()) {
      for (auto const& unit_quaternion : Rotation::paramsExamples()) {
        examples.push_back(params(scaling_factors, unit_quaternion));
      }
    }
    return examples;
  }

  static auto invalidParamsExamples() -> std::vector<Params> {
    return std::vector<Params>({
        Params::Zero(),
        -Params::Ones(),
        -Params::UnitX(),
    });
  }

 private:
  static auto unitQuaternion(Params const& params) -> Eigen::Vector<Scalar, 4> {
    return params.template tail<4>();
  }

  static auto scaleFactors(Params const& params) -> Eigen::Vector<Scalar, 3> {
    return params.template head<3>();
  }

  static auto params(
      Eigen::Vector<Scalar, 3> const& scaling_factors,
      Eigen::Vector<Scalar, 4> const& unit_quaternion) -> Params {
    Params params;
    params.template head<3>() = scaling_factors;
    params.template tail<4>() = unit_quaternion;
    return params;
  }

  static auto angleTimesAxis(Tangent const& tangent) -> Point {
    return tangent.template head<3>();
  }

  static auto logFactors(Tangent const& tangent) -> Point {
    return tangent.template tail<3>();
  }

  static auto tangent(
      Eigen::Vector<Scalar, 3> const& angle_times_axis,
      Eigen::Vector<Scalar, 3> const& log_factors) -> Tangent {
    Tangent tangent;
    tangent.template head<3>() = angle_times_axis;
    tangent.template tail<3>() = log_factors;
    return tangent;
  }
};

}  // namespace lie
}  // namespace sophus
