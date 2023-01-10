// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/calculus/interval.h"

#include "sophus/image/image.h"

#include <gtest/gtest.h>

using namespace sophus;

template <typename T>
void typeChecks() {
  // Check default construction
  Interval<T> mm;
  CHECK(mm.min() == std::numeric_limits<T>::max());
  CHECK(mm.max() == std::numeric_limits<T>::lowest());

  mm = Interval<T>(10, 22);
  CHECK(mm == Interval<T>(10, 22));
  mm.extend(11);
  CHECK(mm == Interval<T>(10, 22));
  mm.extend(190);
  CHECK(mm == Interval<T>(10, 190));
  mm.extend(7);
  CHECK(mm == Interval<T>(7, 190));

  mm.extend(Interval<T>(70, 200));
  CHECK(mm == Interval<T>(7, 200));

  mm.extend(Interval<T>(6, 201));
  CHECK(mm == Interval<T>(6, 201));

  CHECK(mm != Interval<T>(5, 201));
  CHECK(mm != Interval<T>(6, 200));
}

TEST(Interval, scalar_test) {
  typeChecks<double>();
  typeChecks<float>();
  typeChecks<int16_t>();
  typeChecks<uint8_t>();
  typeChecks<size_t>();
  Interval<double> mm(-1.0);
}

TEST(Interval, image_reduction) {
  {
    sophus::MutImage<double> img({10, 10});
    for (int y = 0; y < img.height(); ++y) {
      for (int x = 0; x < img.height(); ++x) {
        img.uncheckedMut(x, y) = x + y * img.width();
      }
    }
    auto min_max = finiteMinMax(img);
    CHECK(min_max == Interval<double>(0.0, img.width() * img.height() - 1.0));
  }
}
