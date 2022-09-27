// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/geometry.pb.h"
#include "sophus/geometry/ray.h"

#include <farm_ng/core/logging/expected.h>

namespace sophus {

farm_ng::Expected<sophus::UnitVector3F64> fromProto(
    proto::UnitVec3F64 const& proto);
proto::UnitVec3F64 toProto(sophus::UnitVector3F64 const& uvec);

farm_ng::Expected<Eigen::Hyperplane<double, 3>> fromProto(
    proto::Hyperplane3F64 const& proto);
proto::Hyperplane3F64 toProto(Eigen::Hyperplane<double, 3> const& plane);

}  // namespace sophus
