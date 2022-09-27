// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image.pb.h"
#include "sophus/image/image_size.h"

#include <farm_ng/core/logging/expected.h>

namespace sophus {

sophus::ImageSize fromProto(proto::ImageSize const& proto);
proto::ImageSize toProto(sophus::ImageSize const& image_size);

}  // namespace sophus
