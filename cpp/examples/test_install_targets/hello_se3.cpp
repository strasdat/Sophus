// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <sophus/lie/se3.h>

#include <iostream>

int main() {
  // Example of create a rigid transformation from an SO(3) = 3D rotation and a
  // translation 3-vector:

  // Let use assume there is a camera in the world. First we describe its
  // orientation in the world reference frame.
  sophus::Rotation3F64 world_from_camera_rotation =
      sophus::Rotation3F64::fromRx(sophus::kPi<double> / 4);
  // Then the position of the camera in the world.
  Eigen::Vector3d camera_in_world(0.0, 0.0, 1.0);

  // The pose (position and orientation) of the camera in the world is
  // constructed by its orientation ``world_from_camera_rotation`` as well as
  // its position ``camera_in_world``.
  sophus::Isometry3F64 world_anchored_camera_pose(
      world_from_camera_rotation, camera_in_world);

  // SE(3) naturally representation is a 4x4 matrix which can be accessed using
  // the .matrix() method:
  std::cout << "world_anchored_camera_pose:\n"
            << world_anchored_camera_pose.matrix() << std::endl;
}
