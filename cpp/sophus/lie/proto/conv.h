// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/lie.pb.h"
#include "sophus/lie/se2.h"
#include "sophus/lie/se3.h"

namespace sophus {

Eigen::Quaterniond fromProto(proto::QuaternionF64 const& proto);
proto::QuaternionF64 toProto(Eigen::Quaterniond const& quat);

So2F64 fromProto(proto::So2F64 const& proto);
proto::So2F64 toProto(sophus::So2F64 const& rotation);

Se2F64 fromProto(proto::Se2F64 const& proto);
proto::Se2F64 toProto(Se2F64 const& pose);

Expected<So3F64> fromProto(proto::So3F64 const& proto);
proto::So3F64 toProto(sophus::So3F64 const& rotation);

Expected<Se3F64> fromProto(proto::Se3F64 const& proto);
proto::Se3F64 toProto(Se3F64 const& pose);

}  // namespace sophus
