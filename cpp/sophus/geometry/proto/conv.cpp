// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/geometry/proto/conv.h"

#include "sophus/linalg/proto/conv.h"

namespace sophus {

Expected<UnitVector3F64> fromProto(proto::UnitVec3F64 const& proto) {
  return UnitVector3F64::tryFromUnitVector(fromProto(proto.vec3()));
}

proto::UnitVec3F64 toProto(sophus::UnitVector3F64 const& uvec) {
  proto::UnitVec3F64 proto;
  *proto.mutable_vec3() = toProto(uvec.params());
  return proto;
}

Expected<Eigen::Hyperplane<double, 3>> fromProto(
    proto::Hyperplane3F64 const& proto) {
  SOPHUS_TRY(sophus::UnitVector3F64, normal, fromProto(proto.normal()));
  return Eigen::Hyperplane<double, 3>{normal.params(), proto.offset()};
}

proto::Hyperplane3F64 toProto(Eigen::Hyperplane<double, 3> const& plane) {
  proto::Hyperplane3F64 proto;
  *proto.mutable_normal() =
      toProto(sophus::UnitVector3F64::fromVectorAndNormalize(plane.normal()));
  proto.set_offset(plane.offset());
  return proto;
}

}  // namespace sophus
