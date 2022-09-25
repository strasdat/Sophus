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

namespace farm_ng {

Expected<sophus::CameraModel> fromProto(proto::CameraModel const& proto) {
  auto get_params = [&proto]() -> Eigen::VectorXd {
    Eigen::VectorXd params(proto.params_size());
    for (int i = 0; i < params.rows(); ++i) {
      params[i] = proto.params(i);
    }
    return params;
  };

  sophus::CameraTransformType model = sophus::CameraTransformType::pinhole;
  if (trySetFromString(model, proto.transform_type())) {
    FARM_ERROR("transform type not supported: {}", proto.transform_type());
  }

  return sophus::CameraModel(
      fromProto(proto.image_size()), model, get_params());
}

proto::CameraModel toProto(sophus::CameraModel const& camera_model) {
  proto::CameraModel proto;
  *proto.mutable_image_size() = toProto(camera_model.imageSize());
  proto.set_transform_type(toString(camera_model.transformType()));
  Eigen::VectorXd params = camera_model.params();
  for (int i = 0; i < params.rows(); ++i) {
    proto.add_params(params[i]);
  }
  return proto;
}

Expected<std::vector<sophus::CameraModel>> fromProto(
    proto::CameraModels const& proto) {
  std::vector<sophus::CameraModel> models;
  for (int i = 0; i < proto.camera_models_size(); ++i) {
    FARM_TRY(sophus::CameraModel cam, fromProto(proto.camera_models(i)));
    models.push_back(cam);
  }
  return models;
}

proto::CameraModels toProto(
    std::vector<sophus::CameraModel> const& camera_models) {
  proto::CameraModels proto;
  for (auto const& model : camera_models) {
    *proto.add_camera_models() = toProto(model);
  }
  return proto;
}

}  // namespace farm_ng
