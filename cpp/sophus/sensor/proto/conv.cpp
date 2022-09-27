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

farm_ng::Expected<CameraModel> fromProto(proto::CameraModel const& proto) {
  auto get_params = [&proto]() -> Eigen::VectorXd {
    Eigen::VectorXd params(proto.paramsSize());
    for (int i = 0; i < params.rows(); ++i) {
      params[i] = proto.params(i);
    }
    return params;
  };

  CameraTransformType model = CameraTransformType::pinhole;
  if (trySetFromString(model, proto.transformType())) {
    FARM_ERROR("transform type not supported: {}", proto.transformType());
  }

  return CameraModel(fromProto(proto.imageSize()), model, get_params());
}

proto::CameraModel toProto(CameraModel const& camera_model) {
  proto::CameraModel proto;
  *proto.mutableImageSize() = toProto(camera_model.imageSize());
  proto.setTransformType(toString(camera_model.transformType()));
  Eigen::VectorXd params = camera_model.params();
  for (int i = 0; i < params.rows(); ++i) {
    proto.addParams(params[i]);
  }
  return proto;
}

farm_ng::Expected<std::vector<CameraModel>> fromProto(
    proto::CameraModels const& proto) {
  std::vector<CameraModel> models;
  for (int i = 0; i < proto.cameraModelsSize(); ++i) {
    FARM_TRY(CameraModel cam, fromProto(proto.cameraModels(i)));
    models.push_back(cam);
  }
  return models;
}

proto::CameraModels toProto(std::vector<CameraModel> const& camera_models) {
  proto::CameraModels proto;
  for (auto const& model : camera_models) {
    *proto.addCameraModels() = toProto(model);
  }
  return proto;
}

}  // namespace sophus
