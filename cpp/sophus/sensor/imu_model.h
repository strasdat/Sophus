// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/enum.h"
#include "sophus/lie/se3.h"

#include <Eigen/Dense>

#include <variant>

namespace sophus {

template <class TT>
auto nonOrthogonalityMatrix(Eigen::Matrix<TT, 3, 1> const& non_orthogonality)
    -> Eigen::Matrix<TT, 3, 3> {
  Eigen::Matrix<TT, 3, 3> lower_triagonal = Eigen::Matrix<TT, 3, 3>::Identity();
  lower_triagonal(1, 0) = non_orthogonality[0];
  lower_triagonal(2, 0) = non_orthogonality[1];
  lower_triagonal(2, 1) = non_orthogonality[2];
  return lower_triagonal;
}

// following: https://timohinzmann.com/publications/icra_2016_rehder.pdf
template <class TT>
struct ScalingNonOrthogonalityGyroModel {
  ScalingNonOrthogonalityGyroModel(
      Eigen::Matrix<TT, 3, 1> const& scale = Eigen::Matrix<TT, 3, 1>::Ones(),
      Eigen::Matrix<TT, 3, 1> const& non_orthogonality =
          Eigen::Matrix<TT, 3, 1>::Zero(),
      Eigen::Matrix<TT, 3, 1> const& gyro_bias =
          Eigen::Matrix<TT, 3, 1>::Zero())
      : scale(scale),
        non_orthogonality(non_orthogonality),
        gyro_bias(gyro_bias) {}

  static auto fromParams(Eigen::Matrix<TT, 9, 1> const& params)
      -> ScalingNonOrthogonalityGyroModel<TT> {
    return ScalingNonOrthogonalityGyroModel<TT>(
        {params[0], params[1], params[2]},
        {params[3], params[4], params[5]},
        {params[6], params[7], params[8]});
  }

  [[nodiscard]] auto gyroMeasurement(
      Eigen::Matrix<TT, 3, 1> const& imu_angular_rate_imu) const
      -> Eigen::Matrix<TT, 3, 1> {
    return Eigen::DiagonalMatrix<TT, 3>(scale) *
               nonOrthogonalityMatrix(non_orthogonality) *
               imu_angular_rate_imu +
           gyro_bias;
  }

  [[nodiscard]] auto params() const -> Eigen::Matrix<TT, 9, 1> {
    Eigen::Matrix<TT, 9, 1> params;
    params << scale[0], scale[1], scale[2], non_orthogonality[0],
        non_orthogonality[1], non_orthogonality[2], gyro_bias[0], gyro_bias[1],
        gyro_bias[2];
    return params;
  }

  Eigen::Matrix<TT, 3, 1> scale;
  Eigen::Matrix<TT, 3, 1> non_orthogonality;
  Eigen::Matrix<TT, 3, 1> gyro_bias;
};

// following: https://timohinzmann.com/publications/icra_2016_rehder.pdf
template <class TT>
struct ScalingNonOrthogonalityAcceleroModel {
  ScalingNonOrthogonalityAcceleroModel(
      Eigen::Matrix<TT, 3, 1> const& scale = Eigen::Matrix<TT, 3, 1>::Ones(),
      Eigen::Matrix<TT, 3, 1> const& non_orthogonality =
          Eigen::Matrix<TT, 3, 1>::Zero(),
      Eigen::Matrix<TT, 3, 1> const& accel_bias =
          Eigen::Matrix<TT, 3, 1>::Zero())
      : scale(scale),
        non_orthogonality(non_orthogonality),
        accel_bias(accel_bias) {}

  static auto fromParams(Eigen::Matrix<TT, 9, 1> const& params)
      -> ScalingNonOrthogonalityAcceleroModel<TT> {
    return ScalingNonOrthogonalityAcceleroModel<TT>(
        {params[0], params[1], params[2]},
        {params[3], params[4], params[5]},
        {params[6], params[7], params[8]});
  }

  [[nodiscard]] auto acceleroMeasurement(
      Eigen::Matrix<TT, 3, 1> const& imu_acceleration_imu) const
      -> Eigen::Matrix<TT, 3, 1> {
    return Eigen::DiagonalMatrix<TT, 3>(scale) *
               nonOrthogonalityMatrix(non_orthogonality) *
               imu_acceleration_imu +
           accel_bias;
  }

  [[nodiscard]] auto params() const -> Eigen::Matrix<TT, 9, 1> {
    Eigen::Matrix<TT, 9, 1> params;
    params << scale[0], scale[1], scale[2], non_orthogonality[0],
        non_orthogonality[1], non_orthogonality[2], accel_bias[0],
        accel_bias[1], accel_bias[2];
    return params;
  }

  Eigen::Matrix<TT, 3, 1> scale;
  Eigen::Matrix<TT, 3, 1> non_orthogonality;
  Eigen::Matrix<TT, 3, 1> accel_bias;
};

SOPHUS_ENUM(GyroModelType, (scaling_non_orthogonality));

SOPHUS_ENUM(AcceleroModelType, (scaling_non_orthogonality));

using GyroModelVariant = std::variant<ScalingNonOrthogonalityGyroModel<double>>;
using AcceleroModelVariant =
    std::variant<ScalingNonOrthogonalityAcceleroModel<double>>;

auto getModelFromType(GyroModelType model_type, Eigen::VectorXd const& params)
    -> GyroModelVariant;

auto getModelFromType(
    AcceleroModelType model_type, Eigen::VectorXd const& params)
    -> AcceleroModelVariant;

static_assert(
    std::variant_size_v<GyroModelVariant> == getCount(GyroModelType()),
    "When the variant GyroModelVariant is updated, one needs to "
    "update the enum GyroModelType as well, and vice versa.");

static_assert(
    std::variant_size_v<AcceleroModelVariant> == getCount(AcceleroModelType()),
    "When the variant AcceleroModelVariant is updated, one needs to "
    "update the enum AcceleroModelType as well, and vice versa.");

class ImuModel {
 public:
  ImuModel(
      GyroModelVariant const& gyro_model,
      AcceleroModelVariant const& accelero_model)
      : gyro_model_(gyro_model), accelero_model_(accelero_model) {}

  auto gyroMeasurement(Eigen::Vector3d const& world_velocity_imu)
      -> Eigen::Vector3d;

  auto acceleroMeasurement(Eigen::Vector3d const& world_acceleration_imu)
      -> Eigen::Vector3d;

  auto gyroModel() -> GyroModelVariant& { return gyro_model_; }

  [[nodiscard]] auto gyroModel() const -> GyroModelVariant const& {
    return gyro_model_;
  }

  auto acceleroModel() -> AcceleroModelVariant& { return accelero_model_; }

  [[nodiscard]] auto acceleroModel() const -> AcceleroModelVariant const& {
    return accelero_model_;
  }

  [[nodiscard]] auto gyroParams() const -> Eigen::VectorXd;

  [[nodiscard]] auto acceleroParams() const -> Eigen::VectorXd;

  [[nodiscard]] auto gyroModelType() const -> GyroModelType;

  [[nodiscard]] auto acceleroModelType() const -> AcceleroModelType;

 private:
  GyroModelVariant gyro_model_;
  AcceleroModelVariant accelero_model_;
};

}  // namespace sophus
