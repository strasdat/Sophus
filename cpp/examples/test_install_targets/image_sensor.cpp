// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <farm_ng/core/logging/logger.h>
#include <sophus/image/runtime_image.h>
#include <sophus/sensor/camera_model.h>

using namespace sophus;

int main() {
  const ImageSize size64{6, 4};
  MutImage<float> mut_image(size64);
  mut_image.fill(0.5f);
  Image<float> image(std::move(mut_image));
  AnyImage<> any_image(image);

  std::vector<CameraModel> camera_models;
  CameraModel pinhole = CameraModel::createDefaultPinholeModel({640, 480});
  Eigen::VectorXd get_params(8);
  get_params << 1000, 1000, 320, 280, 0.1, 0.01, 0.001, 0.0001;
  CameraModel kb3 = CameraModel(
      {640, 480}, CameraDistortionType::kannala_brandt_k3, get_params);

  camera_models.push_back(pinhole);
  camera_models.push_back(kb3);
}
