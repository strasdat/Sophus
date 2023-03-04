// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/lie.pb.h"
#include "sophus/lie/isometry2.h"
#include "sophus/lie/isometry3.h"
#include "sophus/lie/rotation2.h"
#include "sophus/lie/rotation3.h"

namespace sophus {

QuaternionF64 fromProto(proto::QuaternionF64 const& proto);
proto::QuaternionF64 toProto(QuaternionF64 const& quat);

Rotation2<double> fromProto(proto::Rotation2F64 const& proto);
proto::Rotation2F64 toProto(Rotation2<double> const& rotation);

Isometry2<double> fromProto(proto::Isometry2F64 const& proto);
proto::Isometry2F64 toProto(Isometry2<double> const& pose);

Expected<Rotation3<double>> fromProto(proto::Rotation3F64 const& proto);
proto::Rotation3F64 toProto(Rotation3<double> const& rotation);

Expected<Isometry3<double>> fromProto(proto::Isometry3F64 const& proto);
proto::Isometry3F64 toProto(Isometry3<double> const& pose);

}  // namespace sophus
