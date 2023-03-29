// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/group_manifold.h"

#include "sophus/lie/isometry3.h"
#include "sophus/manifold/product_manifold.h"
#include "sophus/manifold/vector_manifold.h"

#include <gtest/gtest.h>

using namespace sophus;

TEST(group_manifold, unit) {
  static_assert(concepts::Manifold<LeftPlus<Isometry3F64>>);

  using Product = ProductManifold<
      LeftPlus<Isometry3F64>,
      UnitVector2F64,
      VectorManifold<double, 2>>;
  static_assert(concepts::BaseManifold<Product>);

  {
    LeftPlus<Isometry3F64> g1(Isometry3F64::fromRx(0.1));
    Eigen::Vector<double, 6> t(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
    auto g2 = g1.oplus(t);

    Eigen::Vector<double, 6> t_proof = g2.ominus(g1);
    FARM_ASSERT_NEAR(t, t_proof, 0.001);
  }

  {
    RightPlus<Isometry3F64> g1(Isometry3F64::fromRx(0.1));
    Eigen::Vector<double, 6> t(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
    auto g2 = g1.oplus(t);

    Eigen::Vector<double, 6> t_proof = g2.ominus(g1);
    FARM_ASSERT_NEAR(t, t_proof, 0.001);
  }
}
