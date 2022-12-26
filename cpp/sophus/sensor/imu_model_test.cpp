// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/sensor/imu_model.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(imu_model, smoke) {
  Eigen::Matrix<double, 3, 1> imu_vel{1.1, 2.0, -3.5};
  Eigen::Matrix<double, 3, 1> imu_acc{11.5, 18.0, -11.0};

  Eigen::Matrix<double, 3, 1> gyr_scale{1.5, 0.2, 3};
  Eigen::Matrix<double, 3, 1> gyr_non_ortho{-0.1, 0.2, 0.3};
  Eigen::Matrix<double, 3, 1> gyr_bias{-0.9, 2.1, 0.1};

  Eigen::Matrix<double, 3, 1> acc_scale{1.1, 0.1, 2.3};
  Eigen::Matrix<double, 3, 1> acc_non_ortho{0.1, -0.2, 0.3};
  Eigen::Matrix<double, 3, 1> acc_bias{1.5, 2, -10};

  Eigen::Matrix<double, 3, 1> vel_meas{0.75, 2.478, -7.94};
  Eigen::Matrix<double, 3, 1> acc_meas{14.15, 3.915, -28.17};

  double tolerance = 0.000001;

  ScalingNonOrthogonalityGyroModel gyro_model =
      ScalingNonOrthogonalityGyroModel(gyr_scale, gyr_non_ortho, gyr_bias);

  ScalingNonOrthogonalityAcceleroModel accel_model =
      ScalingNonOrthogonalityAcceleroModel(acc_scale, acc_non_ortho, acc_bias);

  ImuModel imu_model = ImuModel(gyro_model, accel_model);

  SOPHUS_ASSERT_EQ(
      gyro_model.gyroMeasurement(imu_vel), imu_model.gyroMeasurement(imu_vel));
  SOPHUS_ASSERT_EQ(
      accel_model.acceleroMeasurement(imu_acc),
      imu_model.acceleroMeasurement(imu_acc));

  SOPHUS_ASSERT_NEAR(vel_meas, imu_model.gyroMeasurement(imu_vel), tolerance);
  SOPHUS_ASSERT_NEAR(
      acc_meas, imu_model.acceleroMeasurement(imu_acc), tolerance);
}
