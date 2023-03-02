// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/linalg/homogeneous.h"

#include "sophus/linalg/vector_space.h"

#include <gtest/gtest.h>

using namespace sophus;

template <class TScalar, int kDim>
void unprojProjRountripTest() {
  for (auto const& in_point : pointExamples<TScalar, kDim>()) {
    Eigen::Vector<TScalar, kDim + 1> hpoint = unproj(in_point);
    Eigen::Vector<TScalar, kDim> out_point = proj(hpoint);

    SOPHUS_ASSERT_NEAR(in_point, out_point, 0.001);
  }
}

TEST(homogeneous, unit) {
  unprojProjRountripTest<float, 2>();
  unprojProjRountripTest<float, 3>();
  unprojProjRountripTest<double, 2>();
  unprojProjRountripTest<double, 3>();
}
