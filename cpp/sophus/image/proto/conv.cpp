// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/proto/conv.h"

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

sophus::ImageLayout fromProto(proto::ImageLayout const& proto) {
  return sophus::ImageLayout(fromProto(proto.size()), proto.pitch_bytes());
}

proto::ImageLayout toProto(sophus::ImageLayout const& layout) {
  proto::ImageLayout proto;
  *proto.mutable_size() = toProto(layout.imageSize());
  proto.set_pitch_bytes(layout.pitchBytes());
  return proto;
}

Expected<sophus::PixelFormat> fromProto(proto::PixelFormat const& proto) {
  sophus::PixelFormat format;
  SOPHUS_ASSERT_OR_ERROR(
      trySetFromString(format.number_type, proto.number_type()));
  format.num_components = proto.num_components();
  format.num_bytes_per_component = proto.num_bytes_per_component();
  return format;
}

proto::PixelFormat toProto(sophus::PixelFormat const& layout) {
  proto::PixelFormat proto;
  proto.set_number_type(toString(layout.number_type));
  proto.set_num_components(layout.num_components);
  proto.set_num_bytes_per_component(layout.num_bytes_per_component);
  return proto;
}

Expected<sophus::AnyImage<>> fromProto(proto::DynImage const& proto) {
  SOPHUS_TRY(PixelFormat, format, fromProto(proto.pixel_format()));
  auto layout = fromProto(proto.layout());

  SOPHUS_ASSERT_EQ(size_t(layout.sizeBytes()), proto.data().size());
  SOPHUS_TRY(
      auto, mut_image, sophus::MutAnyImage<>::tryFromFormat(layout, format));
  std::memcpy(mut_image.rawMutPtr(), proto.data().data(), proto.data().size());
  return mut_image;
}

Expected<sophus::IntensityImage<>> intensityImageFromProto(
    proto::DynImage const& proto) {
  SOPHUS_TRY(sophus::AnyImage<>, any_image, fromProto(proto));
  return sophus::IntensityImage<>::tryFrom(any_image);
}

}  // namespace sophus
