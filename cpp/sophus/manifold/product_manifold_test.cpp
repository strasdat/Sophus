// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/manifold/product_manifold.h"

#include "sophus/manifold/unit_vector.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(product_manifold, unit) {
  using Product = ProductManifold<UnitVector3F64, UnitVector2F64>;

  static_assert(concepts::BaseManifold<Product>);
}
