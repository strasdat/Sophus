// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/sensor/proto/conv.h"

#include "sophus/image/proto/conv.h"

namespace sophus {

Expected<CameraModel> fromProto(proto::CameraModel const& proto) {
  auto get_params = [&proto]() -> Eigen::VectorXd {
    Eigen::VectorXd params(proto.params_size());
    for (int i = 0; i < params.rows(); ++i) {
      params[i] = proto.params(i);
    }
    return params;
  };

  CameraDistortionType model = CameraDistortionType::pinhole;
  SOPHUS_ASSERT_OR_ERROR(
      trySetFromString(model, proto.distortion_type()),
      "distortion type not supported: {}",
      proto.distortion_type());

  return CameraModel(fromProto(proto.image_size()), model, get_params());
}

proto::CameraModel toProto(CameraModel const& camera_model) {
  proto::CameraModel proto;
  *proto.mutable_image_size() = toProto(camera_model.imageSize());
  proto.set_distortion_type(toString(camera_model.distortionType()));
  Eigen::VectorXd params = camera_model.params();
  for (int i = 0; i < params.rows(); ++i) {
    proto.add_params(params[i]);
  }
  return proto;
}

Expected<std::vector<CameraModel>> fromProto(proto::CameraModels const& proto) {
  std::vector<CameraModel> models;
  for (int i = 0; i < proto.camera_models_size(); ++i) {
    SOPHUS_TRY(CameraModel, cam, fromProto(proto.camera_models(i)));
    models.push_back(cam);
  }
  return models;
}

proto::CameraModels toProto(std::vector<CameraModel> const& camera_models) {
  proto::CameraModels proto;
  for (auto const& model : camera_models) {
    *proto.add_camera_models() = toProto(model);
  }
  return proto;
}

}  // namespace sophus
