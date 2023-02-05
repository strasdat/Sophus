// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/mut_image_view.h"

#include "sophus/image/image_types.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(MutImageView, empty) {
  MutImageView<float> mut_view;
  SOPHUS_ASSERT(mut_view.isEmpty());
  SOPHUS_ASSERT_EQ(mut_view.imageSize(), ImageSize(0, 0));
  SOPHUS_ASSERT_EQ(mut_view.layout().pitchBytes(), 0u);
  SOPHUS_ASSERT(mut_view.ptrMut() == nullptr);

  MutImageView<float> mut_view2;
  mut_view.copyDataFrom(mut_view2);
  SOPHUS_ASSERT(mut_view2.isEmpty());
}

TEST(MutImageView, create_and_access) {
  // clang-format off
  const std::array<uint16_t, 12> data_u16 =
      { 0,  1,  2,
       10, 11, 12,
       20, 21, 22,
       30, 31, 32};
  // clang-format on
  // clang-format off
    std::array<uint16_t, 12> mut_data_u16 =
       { 0,  0,  0,
         0,  0,  0,
         0,  0,  0,
         0,  0,  0};
  // clang-format on
  ImageSize image_size(3, 2);
  ImageView<uint16_t> view(image_size, data_u16.data());
  MutImageView<uint16_t> mut_view(image_size, mut_data_u16.data());
  mut_view.copyDataFrom(view);

  SOPHUS_ASSERT_IMAGE_EQ(view, mut_view);

  mut_view.fill(111);
  for (int v = 0; v < image_size.height; ++v) {
    for (int u = 0; u < image_size.width; ++u) {
      SOPHUS_ASSERT_EQ(mut_view(u, v), 111);
    }
  }

  ImageSize col_view_size(1, 2);
  MutImageView<uint16_t> mut_col1 = mut_view.mutSubview({1, 0}, col_view_size);
  SOPHUS_ASSERT_EQ(
      mut_view.layout().pitchBytes(), mut_col1.layout().pitchBytes());

  mut_col1.fill(222);

  for (int v = 0; v < image_size.height; ++v) {
    for (int u = 0; u < image_size.width; ++u) {
      if (u == 1) {
        SOPHUS_ASSERT_EQ(mut_view(u, v), 222);
      } else {
        SOPHUS_ASSERT_EQ(mut_view(u, v), 111);
      }
    }
  }
  {
    ImageSize image_size(1, 2);
    using Pixel2U16 = Eigen::Matrix<uint16_t, 2, 1>;
    MutImageView<Pixel2U16> mut_view(
        ImageLayout::makeFromSizeAndPitch<Pixel2U16>(
            image_size,
            image_size.width * sizeof(Pixel2U16) + sizeof(uint16_t)),
        (Pixel2U16*)data_u16.data());
    SOPHUS_ASSERT_EQ(mut_view.imageSize(), image_size);
    SOPHUS_ASSERT_EQ(mut_view.layout().pitchBytes(), 6);
    SOPHUS_ASSERT_EQ(mut_view(0, 0), Pixel2U16(0, 1));
    SOPHUS_ASSERT_EQ(mut_view(0, 1), Pixel2U16(10, 11));
  }
}
