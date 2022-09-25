// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "protos/sophus/linalg.pb.h"

#include <Eigen/Core>
#include <farm_ng/core/logging/expected.h>

namespace farm_ng {

Eigen::Matrix<uint32_t, 2, 1> fromProto(proto::Vec2I64 const& proto);
proto::Vec2I64 toProto(Eigen::Matrix<uint32_t, 2, 1> const& v);

Eigen::Vector2f fromProto(proto::Vec2F32 const& proto);
proto::Vec2F32 toProto(Eigen::Vector2f const& v);

Eigen::Vector2d fromProto(proto::Vec2F64 const& proto);
proto::Vec2F64 toProto(Eigen::Vector2d const& v);

Eigen::Matrix<uint32_t, 3, 1> fromProto(proto::Vec3I64 const& proto);
proto::Vec3I64 toProto(Eigen::Matrix<uint32_t, 3, 1> const& v);

Eigen::Vector3f fromProto(proto::Vec3F32 const& proto);
proto::Vec3F32 toProto(Eigen::Vector3f const& v);

Eigen::Vector3d fromProto(proto::Vec3F64 const& proto);
proto::Vec3F64 toProto(Eigen::Vector3d const& v);

}  // namespace farm_ng
