// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image.pb.h"
#include "sophus/image/dyn_image_types.h"

namespace sophus {

sophus::ImageSize fromProto(proto::ImageSize const& proto);
proto::ImageSize toProto(sophus::ImageSize const& image_size);

sophus::ImageLayout fromProto(proto::ImageLayout const& proto);
proto::ImageLayout toProto(sophus::ImageLayout const& layout);

Expected<sophus::PixelFormat> fromProto(proto::PixelFormat const& proto);
proto::PixelFormat toProto(sophus::PixelFormat const& layout);

Expected<sophus::AnyImage<>> fromProto(proto::DynImage const& proto);
Expected<sophus::IntensityImage<>> intensityImageFromProto(
    proto::DynImage const& proto);

template <class TPredicate>
proto::DynImage toProto(sophus::DynImage<TPredicate> const& image) {
  proto::DynImage proto;
  *proto.mutable_layout() = toProto(image.layout());
  *proto.mutable_pixel_format() = toProto(image.pixelFormat());
  proto.set_data(
      std::string(image.rawPtr(), image.rawPtr() + image.sizeBytes()));

  SOPHUS_ASSERT_EQ(size_t(image.layout().sizeBytes()), proto.data().size());

  return proto;
}
}  // namespace sophus
