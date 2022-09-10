// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/lie/se3.h"

#include <Eigen/Dense>
#include <farm_ng/core/enum/enum.h>

#include <variant>

namespace sophus {

template <class TT>
Eigen::Matrix<TT, 3, 3> nonOrthogonalityMatrix(
    const Eigen::Matrix<TT, 3, 1>& non_orthogonality) {
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
      const Eigen::Matrix<TT, 3, 1>& scale = Eigen::Matrix<TT, 3, 1>::Ones(),
      const Eigen::Matrix<TT, 3, 1>& non_orthogonality =
          Eigen::Matrix<TT, 3, 1>::Zero(),
      const Eigen::Matrix<TT, 3, 1>& gyro_bias =
          Eigen::Matrix<TT, 3, 1>::Zero())
      : scale(scale),
        non_orthogonality(non_orthogonality),
        gyro_bias(gyro_bias) {}

  static ScalingNonOrthogonalityGyroModel<TT> fromParams(
      const Eigen::Matrix<TT, 9, 1>& params) {
    return ScalingNonOrthogonalityGyroModel<TT>(
        {params[0], params[1], params[2]},
        {params[3], params[4], params[5]},
        {params[6], params[7], params[8]});
  }

  [[nodiscard]] Eigen::Matrix<TT, 3, 1> gyroMeasurement(
      const Eigen::Matrix<TT, 3, 1>& imu_angular_rate_imu) const {
    return Eigen::DiagonalMatrix<TT, 3>(scale) *
               nonOrthogonalityMatrix(non_orthogonality) *
               imu_angular_rate_imu +
           gyro_bias;
  }

  [[nodiscard]] Eigen::Matrix<TT, 9, 1> params() const {
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
      const Eigen::Matrix<TT, 3, 1>& scale = Eigen::Matrix<TT, 3, 1>::Ones(),
      const Eigen::Matrix<TT, 3, 1>& non_orthogonality =
          Eigen::Matrix<TT, 3, 1>::Zero(),
      const Eigen::Matrix<TT, 3, 1>& accel_bias =
          Eigen::Matrix<TT, 3, 1>::Zero())
      : scale(scale),
        non_orthogonality(non_orthogonality),
        accel_bias(accel_bias) {}

  static ScalingNonOrthogonalityAcceleroModel<TT> fromParams(
      const Eigen::Matrix<TT, 9, 1>& params) {
    return ScalingNonOrthogonalityAcceleroModel<TT>(
        {params[0], params[1], params[2]},
        {params[3], params[4], params[5]},
        {params[6], params[7], params[8]});
  }

  [[nodiscard]] Eigen::Matrix<TT, 3, 1> acceleroMeasurement(
      const Eigen::Matrix<TT, 3, 1>& imu_acceleration_imu) const {
    return Eigen::DiagonalMatrix<TT, 3>(scale) *
               nonOrthogonalityMatrix(non_orthogonality) *
               imu_acceleration_imu +
           accel_bias;
  }

  [[nodiscard]] Eigen::Matrix<TT, 9, 1> params() const {
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

FARM_ENUM(GyroModelType, (scaling_non_orthogonality));

FARM_ENUM(AcceleroModelType, (scaling_non_orthogonality));

using GyroModelVariant = std::variant<ScalingNonOrthogonalityGyroModel<double>>;
using AcceleroModelVariant =
    std::variant<ScalingNonOrthogonalityAcceleroModel<double>>;

GyroModelVariant getModelFromType(
    GyroModelType model_type, const Eigen::VectorXd& params);

AcceleroModelVariant getModelFromType(
    AcceleroModelType model_type, const Eigen::VectorXd& params);

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
      const GyroModelVariant& gyro_model,
      const AcceleroModelVariant& accelero_model)
      : gyro_model_(gyro_model), accelero_model_(accelero_model) {}

  Eigen::Vector3d gyroMeasurement(const Eigen::Vector3d& world_velocity_imu);

  Eigen::Vector3d acceleroMeasurement(
      const Eigen::Vector3d& world_acceleration_imu);

  GyroModelVariant& mutGyroModel() { return gyro_model_; }

  [[nodiscard]] const GyroModelVariant& gyroModel() const {
    return gyro_model_;
  }

  AcceleroModelVariant& mutAcceleroModel() { return accelero_model_; }

  [[nodiscard]] const AcceleroModelVariant& acceleroModel() const {
    return accelero_model_;
  }

  [[nodiscard]] Eigen::VectorXd gyroParams() const;

  [[nodiscard]] Eigen::VectorXd acceleroParams() const;

  [[nodiscard]] GyroModelType gyroModelType() const;

  [[nodiscard]] AcceleroModelType acceleroModelType() const;

 private:
  std::string frame_name_;

  GyroModelVariant gyro_model_;
  AcceleroModelVariant accelero_model_;
};

}  // namespace sophus
