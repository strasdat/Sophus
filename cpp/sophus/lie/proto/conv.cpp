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

Eigen::Quaterniond fromProto(proto::QuaternionF64 const& proto) {
  Eigen::Quaterniond quat;
  quat.vec() = fromProto(proto.imag());
  quat.w() = proto.real();
  return quat;
}

proto::QuaternionF64 toProto(Eigen::Quaterniond const& quat) {
  proto::QuaternionF64 proto;
  proto.set_real(quat.w());
  *proto.mutable_imag() = toProto(quat.vec().eval());
  return proto;
}

So2F64 fromProto(proto::So2F64 const& proto) { return So2F64(proto.theta()); }

proto::So2F64 toProto(sophus::So2F64 const& rotation) {
  proto::So2F64 proto;
  proto.set_theta(rotation.log());
  return proto;
}

Se2F64 fromProto(proto::Se2F64 const& proto) {
  return Se2F64(fromProto(proto.so2()), fromProto(proto.translation()));
}

proto::Se2F64 toProto(Se2F64 const& pose) {
  proto::Se2F64 proto;
  *proto.mutable_so2() = toProto(pose.so2());
  *proto.mutable_translation() = toProto(pose.translation());
  return proto;
}

Expected<So3F64> fromProto(proto::So3F64 const& proto) {
  Eigen::Quaterniond quat = fromProto(proto.unit_quaternion());
  static double constexpr kEps = 1e-6;
  if (std::abs(quat.squaredNorm() - 1.0) > kEps) {
    return SOPHUS_UNEXPECTED(
        "quaternion norm ({}) is not close to 1:\n{}",
        quat.squaredNorm(),
        quat.coeffs().transpose());
  }

  return So3F64(quat);
}

proto::So3F64 toProto(sophus::So3F64 const& rotation) {
  proto::So3F64 proto;
  *proto.mutable_unit_quaternion() = toProto(rotation.unitQuaternion());
  return proto;
}

Expected<sophus::Se3F64> fromProto(proto::Se3F64 const& proto) {
  SOPHUS_TRY(sophus::So3F64 so3, fromProto(proto.so3()));
  return sophus::SE3d(so3, fromProto(proto.translation()));
}

proto::Se3F64 toProto(Se3F64 const& pose) {
  proto::Se3F64 proto;
  *proto.mutable_so3() = toProto(pose.so3());
  *proto.mutable_translation() = toProto(pose.translation());
  return proto;
}

}  // namespace sophus
