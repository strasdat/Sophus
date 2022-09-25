// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "protos/sophus/sensor.pb.h"
#include "sophus/sensor/camera_model.h"

#include <farm_ng/core/logging/expected.h>

namespace farm_ng {

Expected<sophus::CameraModel> fromProto(proto::CameraModel const& proto);
proto::CameraModel toProto(sophus::CameraModel const& camera_model);

Expected<std::vector<sophus::CameraModel>> fromProto(
    proto::CameraModels const& proto);
proto::CameraModels toProto(
    std::vector<sophus::CameraModel> const& camera_models);

}  // namespace farm_ng
