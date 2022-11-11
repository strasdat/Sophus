// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/sensor/proto/conv.h"

#include "farm_ng/core/logging/logger.h"
#include "sophus/image/proto/conv.h"

namespace sophus {

farm_ng::Expected<Z1ProjCameraModel> fromProto(
    proto::Z1ProjCameraModel const& proto) {
  auto get_params = [&proto]() -> Eigen::VectorXd {
    Eigen::VectorXd params(proto.params_size());
    for (int i = 0; i < params.rows(); ++i) {
      params[i] = proto.params(i);
    }
    return params;
  };

  Z1ProjDistortionType model = Z1ProjDistortionType::pinhole;
  if (trySetFromString(model, proto.distortion_type())) {
    FARM_ERROR("distortion type not supported: {}", proto.distortion_type());
  }

  return Z1ProjCameraModel(fromProto(proto.image_size()), model, get_params());
}

proto::Z1ProjCameraModel toProto(Z1ProjCameraModel const& camera_model) {
  proto::Z1ProjCameraModel proto;
  *proto.mutable_image_size() = toProto(camera_model.imageSize());
  proto.set_distortion_type(toString(camera_model.distortionType()));
  Eigen::VectorXd params = camera_model.params();
  for (int i = 0; i < params.rows(); ++i) {
    proto.add_params(params[i]);
  }
  return proto;
}

farm_ng::Expected<std::vector<Z1ProjCameraModel>> fromProto(
    proto::Z1ProjCameraModels const& proto) {
  std::vector<Z1ProjCameraModel> models;
  for (int i = 0; i < proto.camera_models_size(); ++i) {
    FARM_TRY(Z1ProjCameraModel cam, fromProto(proto.camera_models(i)));
    models.push_back(cam);
  }
  return models;
}

proto::Z1ProjCameraModels toProto(
    std::vector<Z1ProjCameraModel> const& camera_models) {
  proto::Z1ProjCameraModels proto;
  for (auto const& model : camera_models) {
    *proto.add_camera_models() = toProto(model);
  }
  return proto;
}

}  // namespace sophus
