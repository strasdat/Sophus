// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/proto/conv.h"

#include <gtest/gtest.h>

namespace sophus::test {

template <class TPixel>
void testPixelFormat() {
  auto format_in = PixelFormat::fromTemplate<TPixel>();
  auto maybe_format = fromProto(toProto(format_in));
  PixelFormat format_out = SOPHUS_UNWRAP(maybe_format);
  SOPHUS_ASSERT_EQ(format_in, format_out);
}

template <class TPixel>
void testDynImage(std::vector<TPixel> const& pixels) {
  MutImage<TPixel> mut_image(ImageSize(7, 4));
  size_t i = 0;
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 7; ++x) {
      mut_image.mut(x, y) = pixels[i];
      i = (i + 1) % pixels.size();
    }
  }
  DynImage<> dyn_image_in(std::move(mut_image));
  auto proto = toProto(dyn_image_in);
  auto maybe_dyn_image = fromProto(proto);

  DynImage<> dyn_image_out = SOPHUS_UNWRAP(maybe_dyn_image);
  SOPHUS_ASSERT_IMAGE_EQ(
      dyn_image_in.image<TPixel>(), dyn_image_out.image<TPixel>());
}

template <class TPixel>
void testIntensityImage(std::vector<TPixel> const& pixels) {
  MutImage<TPixel> mut_image(ImageSize(7, 4));
  size_t i = 0;
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 7; ++x) {
      mut_image.mut(x, y) = pixels[i];
      i = (i + 1) % pixels.size();
    }
  }
  IntensityImage<> dyn_image_in(std::move(mut_image));
  auto proto = toProto(dyn_image_in);
  auto maybe_dyn_image = intensityImageFromProto(proto);
  IntensityImage<> dyn_image_out = SOPHUS_UNWRAP(maybe_dyn_image);

  SOPHUS_ASSERT_IMAGE_EQ(
      dyn_image_in.image<TPixel>(), dyn_image_out.image<TPixel>());
}

TEST(conv_image, roundtrip) {
  ImageSize size_in(600, 480);
  ImageSize size_out = fromProto(toProto(size_in));
  SOPHUS_ASSERT_EQ(size_in, size_out);

  ImageLayout layout_in(size_in, 640);
  ImageLayout layout_out = fromProto(toProto(layout_in));
  SOPHUS_ASSERT_EQ(layout_in, layout_out);

  testPixelFormat<float>();
  testPixelFormat<uint16_t>();
  testPixelFormat<uint8_t>();
  testPixelFormat<Eigen::Vector2f>();
  testPixelFormat<Eigen::Vector3<uint16_t>>();

  testDynImage<float>({0.0, 1.0, 0.5});
  testDynImage<uint16_t>({0u, 1111u, 34u});
  testDynImage<Eigen::Vector3f>(
      {Eigen::Vector3f::Ones(),
       Eigen::Vector3f::Zero(),
       Eigen::Vector3f(0.1, 0.2, 0.3)});

  testIntensityImage<float>({0.0, 1.0, 0.5});
  testIntensityImage<uint16_t>({0u, 1111u, 34u});
  testIntensityImage<Eigen::Vector3f>(
      {Eigen::Vector3f::Ones(),
       Eigen::Vector3f::Zero(),
       Eigen::Vector3f(0.1, 0.2, 0.3)});
}
}  // namespace sophus::test
