// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/proto/conv.h"

#include "farm_ng/core/logging/logger.h"

namespace sophus {

sophus::ImageSize fromProto(proto::ImageSize const& proto) {
  sophus::ImageSize image_size;
  image_size.width = proto.width();
  image_size.height = proto.height();
  return image_size;
}

proto::ImageSize toProto(sophus::ImageSize const& image_size) {
  proto::ImageSize proto;
  proto.set_width(image_size.width);
  proto.set_height(image_size.height);
  return proto;
}

}  // namespace sophus
