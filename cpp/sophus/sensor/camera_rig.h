// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/lie/se3.h"
#include "sophus/sensor/camera_model.h"

namespace sophus {

/// Camera as part of a sensor `rig`.
struct CameraInRig {
  CameraInRig() {}
  explicit CameraInRig(CameraModel const& camera_model)
      : camera_model(camera_model) {}

  /// Camera intrinsics
  CameraModel camera_model;

  /// Camera extrinsics
  sophus::SE3d rig_from_camera;
};

/// Sensor rig with multiple cameras.
struct MultiCameraRig {
  std::vector<CameraInRig> cameras_in_rig;

  void transformRig(sophus::SE3d const& new_rig_from_rig) {
    for (auto& camera_in_rig : cameras_in_rig) {
      camera_in_rig.rig_from_camera =
          new_rig_from_rig * camera_in_rig.rig_from_camera;
    }
  }

  void transformRig(uint32_t camera_num) {
    sophus::SE3d new_rig_from_rig =
        cameras_in_rig[camera_num].rig_from_camera.inverse();
    transformRig(new_rig_from_rig);
  }
};

}  // namespace sophus
