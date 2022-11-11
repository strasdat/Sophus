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

farm_ng::Expected<sophus::Z1ProjCameraModel> fromProto(
    proto::Z1ProjCameraModel const& proto);
proto::Z1ProjCameraModel toProto(Z1ProjCameraModel const& camera_model);

farm_ng::Expected<std::vector<Z1ProjCameraModel>> fromProto(
    proto::Z1ProjCameraModels const& proto);
proto::Z1ProjCameraModels toProto(
    std::vector<Z1ProjCameraModel> const& camera_models);

}  // namespace sophus
