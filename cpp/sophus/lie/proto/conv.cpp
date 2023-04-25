// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/proto/conv.h"

#include "sophus/linalg/proto/conv.h"

namespace sophus {

QuaternionF64 fromProto(proto::QuaternionF64 const& proto) {
  QuaternionF64 quat;
  quat.imag() = fromProto(proto.imag());
  quat.real() = proto.real();
  return quat;
}

proto::QuaternionF64 toProto(QuaternionF64 const& quat) {
  proto::QuaternionF64 proto;
  proto.set_real(quat.real());
  *proto.mutable_imag() = toProto(quat.imag().eval());
  return proto;
}

Rotation2F64 fromProto(proto::Rotation2F64 const& proto) {
  return Rotation2F64(proto.theta());
}

proto::Rotation2F64 toProto(sophus::Rotation2F64 const& rotation) {
  proto::Rotation2F64 proto;
  proto.set_theta(rotation.log()[0]);
  return proto;
}

Isometry2F64 fromProto(proto::Isometry2F64 const& proto) {
  return Isometry2F64(
      fromProto(proto.translation()), fromProto(proto.rotation()));
}

proto::Isometry2F64 toProto(Isometry2F64 const& pose) {
  proto::Isometry2F64 proto;
  *proto.mutable_rotation() = toProto(pose.rotation());
  *proto.mutable_translation() = toProto(pose.translation().eval());
  return proto;
}

Expected<Rotation3F64> fromProto(proto::Rotation3F64 const& proto) {
  QuaternionF64 quat = fromProto(proto.unit_quaternion());
  static double constexpr kEps = 1e-6;
  if (std::abs(quat.squaredNorm() - 1.0) > kEps) {
    return SOPHUS_UNEXPECTED(
        "quaternion norm ({}) is not close to 1:\n{}",
        quat.squaredNorm(),
        quat.params().transpose());
  }

  return Rotation3F64::fromUnitQuaternion(quat);
}

proto::Rotation3F64 toProto(sophus::Rotation3F64 const& rotation) {
  proto::Rotation3F64 proto;
  *proto.mutable_unit_quaternion() = toProto(rotation.unitQuaternion());
  return proto;
}

Expected<sophus::Isometry3F64> fromProto(proto::Isometry3F64 const& proto) {
  SOPHUS_TRY(sophus::Rotation3F64, rotation, fromProto(proto.rotation()));
  return Isometry3(fromProto(proto.translation()), rotation);
}

proto::Isometry3F64 toProto(Isometry3F64 const& pose) {
  proto::Isometry3F64 proto;
  *proto.mutable_rotation() = toProto(pose.rotation());
  *proto.mutable_translation() = toProto(pose.translation().eval());
  return proto;
}

}  // namespace sophus
