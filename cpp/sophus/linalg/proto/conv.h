// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include <Eigen/Core>
#include <sophus/linalg.pb.h>

namespace sophus {

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

Eigen::Matrix2f fromProto(proto::Mat2F32 const& proto);
proto::Mat2F32 toProto(Eigen::Matrix2f const& v);

Eigen::Matrix2d fromProto(proto::Mat2F64 const& proto);
proto::Mat2F64 toProto(Eigen::Matrix2d const& v);

Eigen::Matrix3f fromProto(proto::Mat3F32 const& proto);
proto::Mat3F32 toProto(Eigen::Matrix3f const& v);

Eigen::Matrix3d fromProto(proto::Mat3F64 const& proto);
proto::Mat3F64 toProto(Eigen::Matrix3d const& v);

}  // namespace sophus
