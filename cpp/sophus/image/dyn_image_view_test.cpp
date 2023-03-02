// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/dyn_image_types.h"

#include <gtest/gtest.h>

using namespace sophus;

#define SOPHUS_TEST_IMG_EQ(lhs, rhs) /* NOLINT*/        \
  do {                                                  \
    SOPHUS_ASSERT_EQ(lhs.imageSize(), rhs.imageSize()); \
    for (int v = 0; v < lhs.imageSize().height; ++v) {  \
      for (int u = 0; u < lhs.imageSize().width; ++u) { \
        SOPHUS_ASSERT_EQ(lhs(u, v), rhs(u, v));         \
      }                                                 \
    }                                                   \
  } while (false)

auto sum(ImageView<float> view) -> float {
  float s = 0.0;
  for (int v = 0; v < view.height(); ++v) {
    float const* row = view.rowPtr(v);
    for (int u = 0; u < view.width(); ++u) {
      float p = row[u];
      s += p;
    }
  }
  return s;
}

void plusOne(MutImageView<float> mut_view) {
  for (int v = 0; v < mut_view.height(); ++v) {
    float* row = mut_view.rowPtrMut(v);
    for (int u = 0; u < mut_view.width(); ++u) {
      float& p = row[u];
      p += 1.f;
    }
  }
}

TEST(IntensityImageView, subview) {
  MutImage<float> mut_image(ImageSize(4, 4));
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      mut_image.mut(x, y) = 4 * y + x;
    }
  }

  Image<float> ref_image = std::move(mut_image);
  IntensityImage<> dyn_image = ref_image;
  IntensityImageView runtime_sub = dyn_image.subview({1, 1}, {2, 2});
  SOPHUS_ASSERT_EQ(runtime_sub.width(), 2);
  SOPHUS_ASSERT_EQ(runtime_sub.height(), 2);

  {
    ImageView<float> sub = runtime_sub.imageView<float>();

    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        SOPHUS_ASSERT_EQ(sub(x, y), ref_image(x + 1, y + 1));
      }
    }
  }
}

TEST(ClassHierarchy, call_function) {
  // The sum function takes a ImageView<float> as input. Hence we can pass in:
  //  - ImageView<float>
  //  - MutImageView<float>
  //  - MutImage<float>
  //  - Image<float>

  // clang-format off
  std::array<float, 6> data =
      { 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f };
  // clang-format on

  ImageSize image_size(3, 2);
  ImageView<float> view(image_size, data.data());
  float s = sum(view);
  SOPHUS_ASSERT_EQ(s, 3.f);

  MutImageView<float> mut_view(image_size, data.data());
  s = sum(view);
  SOPHUS_ASSERT_EQ(s, 3.f);

  MutImage<float> mut_image(image_size);
  mut_image.fill(0.5f);
  s = sum(mut_image);
  SOPHUS_ASSERT_EQ(s, 3.f);

  Image<float> image = Image<float>::makeCopyFrom(mut_image);
  s = sum(image);
  SOPHUS_ASSERT_EQ(s, 3.f);

  // The plusOne function takes a MutImageView<float> as input. Hence we can
  // pass in:
  //  - MutImageView<float>
  //  - MutImage<float>
  plusOne(mut_view);
  s = sum(mut_view);
  SOPHUS_ASSERT_EQ(s, 9.f);

  plusOne(mut_image);
  s = sum(mut_image);
  SOPHUS_ASSERT_EQ(s, 9.f);

  // won't compile, since ImageView is not a MutImageView:
  // plusOne(view);

  // won't compile, since Image is a ImageView but not a MutImageView:
  // plusOne(image);
}
