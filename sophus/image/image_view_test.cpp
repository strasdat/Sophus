// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/image_view.h"

#include "sophus/image/image_types.h"

#include <farm_ng/core/logging/logger.h>
#include <gtest/gtest.h>

using namespace sophus;

TEST(ImageView, empty) {
  ImageView<float> view;
  FARM_CHECK(view.isEmpty());
  FARM_CHECK_EQ(view.imageSize(), ImageSize(0, 0));
  FARM_CHECK_EQ(view.shape().pitchBytes(), 0u);
  FARM_CHECK(view.data() == nullptr);
}

TEST(ImageView, create_and_access) {
  // clang-format off
  const std::array<uint16_t, 12> data_u16 =
      { 0,  1,  2,
       10, 11, 12,
       20, 21, 22,
       30, 31, 32};
  // clang-format on

  {
    ImageSize image_size(3, 2);
    ImageView<uint16_t> view(image_size, data_u16.data());
    FARM_CHECK_EQ(image_size, view.imageSize());
    FARM_CHECK_EQ(
        image_size.width * sizeof(uint16_t), view.shape().pitchBytes());

    FARM_CHECK(!view.rowInBounds(-1));
    FARM_CHECK(view.rowInBounds(0));
    FARM_CHECK(view.rowInBounds(1));
    FARM_CHECK(!view.rowInBounds(2));

    FARM_CHECK_EQ(view.checked(0, 0), 0);
    FARM_CHECK_EQ(view.checked(0, 1), 10);
    FARM_CHECK_EQ(view.checked(2, 0), 2);

    FARM_CHECK_EQ(size_t(view.data()), size_t(data_u16.data()));

    ImageSize col_view_size(1, 2);
    ImageView<uint16_t> col1 = view.subview({1, 0}, col_view_size);
    FARM_CHECK_EQ(view.shape().pitchBytes(), col1.shape().pitchBytes());

    FARM_CHECK_EQ(col1.checked(0, 0), 1);
    FARM_CHECK_EQ(col1.checked(0, 1), 11);
    FARM_CHECK_EQ(col_view_size, col1.imageSize());
  }

  ImageSize image_size(1, 2);
  using Pixel2U16 = Eigen::Matrix<uint16_t, 2, 1>;

  ImageView<Pixel2U16> view(
      ImageShape::makeFromSizeAndPitch<Pixel2U16>(
          image_size, image_size.width * sizeof(Pixel2U16) + sizeof(uint16_t)),
      (Pixel2U16*)data_u16.data());
  FARM_CHECK_EQ(view.imageSize(), image_size);
  FARM_CHECK_EQ(view.shape().pitchBytes(), 6);

  FARM_CHECK_EQ(view.checked(0, 0), Pixel2U16(0, 1));
  FARM_CHECK_EQ(view.checked(0, 1), Pixel2U16(10, 11));
}

TEST(MutImageView, empty) {
  MutImageView<float> mut_view;
  FARM_CHECK(mut_view.isEmpty());
  FARM_CHECK_EQ(mut_view.imageSize(), ImageSize(0, 0));
  FARM_CHECK_EQ(mut_view.shape().pitchBytes(), 0u);
  FARM_CHECK(mut_view.mutData() == nullptr);

  MutImageView<float> mut_view2;
  mut_view.copyDataFrom(mut_view2);
  FARM_CHECK(mut_view2.isEmpty());
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

  FARM_CHECK_IMAGE_EQ(view, mut_view);

  mut_view.fill(111);
  for (int v = 0; v < image_size.height; ++v) {
    for (int u = 0; u < image_size.width; ++u) {
      FARM_CHECK_EQ(mut_view.checked(u, v), 111);
    }
  }

  ImageSize col_view_size(1, 2);
  MutImageView<uint16_t> mut_col1 = mut_view.mutSubview({1, 0}, col_view_size);
  FARM_CHECK_EQ(mut_view.shape().pitchBytes(), mut_col1.shape().pitchBytes());

  mut_col1.fill(222);

  for (int v = 0; v < image_size.height; ++v) {
    for (int u = 0; u < image_size.width; ++u) {
      if (u == 1) {
        FARM_CHECK_EQ(mut_view.checked(u, v), 222);
      } else {
        FARM_CHECK_EQ(mut_view.checked(u, v), 111);
      }
    }
  }
  {
    ImageSize image_size(1, 2);
    using Pixel2U16 = Eigen::Matrix<uint16_t, 2, 1>;
    MutImageView<Pixel2U16> mut_view(
        ImageShape::makeFromSizeAndPitch<Pixel2U16>(
            image_size,
            image_size.width * sizeof(Pixel2U16) + sizeof(uint16_t)),
        (Pixel2U16*)data_u16.data());
    FARM_CHECK_EQ(mut_view.imageSize(), image_size);
    FARM_CHECK_EQ(mut_view.shape().pitchBytes(), 6);
    FARM_CHECK_EQ(mut_view.checked(0, 0), Pixel2U16(0, 1));
    FARM_CHECK_EQ(mut_view.checked(0, 1), Pixel2U16(10, 11));
  }
}
