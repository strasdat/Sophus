// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "protos/sophus/lie.pb.h"
#include "sophus/lie/se3.h"

#include <farm_ng/core/logging/expected.h>

namespace farm_ng {

Eigen::Quaterniond fromProto(proto::QuaternionF64 const& proto);
proto::QuaternionF64 toProto(Eigen::Quaterniond const& quat);

Expected<sophus::So3F64> fromProto(proto::So3F64 const& proto);
proto::So3F64 toProto(sophus::So3F64 const& rotation);

Expected<sophus::Se3F64> fromProto(proto::Se3F64 const& proto);
proto::Se3F64 toProto(sophus::Se3F64 const& pose);

}  // namespace farm_ng
