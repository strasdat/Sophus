// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/sensor.pb.h"
#include "sophus/sensor/camera_model.h"

#include <farm_ng/core/logging/expected.h>

namespace sophus {

farm_ng::Expected<sophus::CameraModel> fromProto(
    proto::CameraModel const& proto);
proto::CameraModel toProto(CameraModel const& camera_model);

farm_ng::Expected<std::vector<CameraModel>> fromProto(
    proto::CameraModels const& proto);
proto::CameraModels toProto(std::vector<CameraModel> const& camera_models);

}  // namespace sophus
