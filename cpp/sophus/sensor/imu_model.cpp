// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/sensor/imu_model.h"

namespace sophus {

GyroModelVariant getModelFromType(
    GyroModelType model_type, Eigen::VectorXd const& params) {
  switch (model_type) {
    case GyroModelType::scaling_non_orthogonality: {
      return ScalingNonOrthogonalityGyroModel<double>::fromParams(params);
      break;
    }
  }
  FARM_PANIC("logic error");
}

AcceleroModelVariant getModelFromType(
    AcceleroModelType model_type, Eigen::VectorXd const& params) {
  switch (model_type) {
    case AcceleroModelType::scaling_non_orthogonality: {
      return ScalingNonOrthogonalityAcceleroModel<double>::fromParams(params);
      break;
    }
  }
  FARM_PANIC("logic error");
}

Eigen::Vector3d ImuModel::gyroMeasurement(
    Eigen::Vector3d const& world_velocity_imu) {
  return std::visit(
      [&world_velocity_imu](auto&& arg) -> Eigen::Vector3d {
        return arg.gyroMeasurement(world_velocity_imu);
      },
      this->gyro_model_);
}

Eigen::Vector3d ImuModel::acceleroMeasurement(
    Eigen::Vector3d const& world_acceleration_imu) {
  return std::visit(
      [&world_acceleration_imu](auto&& arg) -> Eigen::Vector3d {
        return arg.acceleroMeasurement(world_acceleration_imu);
      },
      this->accelero_model_);
}

GyroModelType ImuModel::gyroModelType() const {
  return std::visit(
      [](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<
                          T,
                          ScalingNonOrthogonalityGyroModel<double>>) {
          return GyroModelType::scaling_non_orthogonality;
        } else {
          static_assert(::sophus::AlwaysFalse<T>, "non-exhaustive visitor!");
        }
      },
      this->gyroModel());
}

Eigen::VectorXd ImuModel::gyroParams() const {
  return std::visit([](auto&& arg) { return arg.params(); }, this->gyroModel());
}

Eigen::VectorXd ImuModel::acceleroParams() const {
  return std::visit(
      [](auto&& arg) { return arg.params(); }, this->acceleroModel());
}

AcceleroModelType ImuModel::acceleroModelType() const {
  return std::visit(
      [](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<
                          T,
                          ScalingNonOrthogonalityAcceleroModel<double>>) {
          return AcceleroModelType::scaling_non_orthogonality;
        } else {
          static_assert(AlwaysFalse<T>, "non-exhaustive visitor!");
        }
      },
      this->acceleroModel());
}

}  // namespace sophus
