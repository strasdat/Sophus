// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/calculus/num_diff.h"
#include "sophus/concepts/group_accessors.h"
#include "sophus/linalg/vector_space.h"

namespace sophus {
namespace test {

template <concepts::accessors::Translation TGroup>
void runTranslationAccessorTests() {
  using Scalar = typename TGroup::Scalar;
  int constexpr k_point_dim = TGroup::kPointDim;
  Eigen::Vector<Scalar, k_point_dim> p;
  for (int i = 0; i < k_point_dim; ++i) {
    p[i] = 0.1 * i;
  }
  SOPHUS_ASSERT_NEAR(TGroup(p).translation(), p, kEpsilon<Scalar>);
}

template <concepts::accessors::Rotation TGroup>
void runRotationAccessorTests() {
  // TODO
}

template <concepts::accessors::Isometry TGroup>
void runIsometryAccessorTests() {
  // TODO
}

template <concepts::accessors::SpiralSimilarity TGroup>
void runSpiralSimilarityAccessorTests() {
  // TODO
}

template <concepts::accessors::Similarity TGroup>
void runSimilarityAccessorTests() {
  // TODO
}

template <concepts::accessors::UnitComplex TGroup>
void runUnitComplexTests() {
  // TODO
}

template <concepts::accessors::UnitQuaternion TGroup>
void runUnitQuaternionTests() {
  // TODO
}

template <concepts::accessors::Rotation2 TGroup>
void runRotation2AccessorTests() {
  runRotationAccessorTests<TGroup>();
  // TODO
}

template <concepts::accessors::Rotation3 TGroup>
void runRotation3AccessorTests() {
  runRotationAccessorTests<TGroup>();
  // TODO
}

template <concepts::accessors::SpiralSimilarity2 TGroup>
void runSpiralSimilarity2AccessorTests() {
  runSpiralSimilarityAccessorTests<TGroup>();
  runRotation2AccessorTests<TGroup>();
  // TODO
}

template <concepts::accessors::SpiralSimilarity3 TGroup>
void runSpiralSimilarity3AccessorTests() {
  runSpiralSimilarityAccessorTests<TGroup>();
  runRotation3AccessorTests<TGroup>();
  // TODO
}

template <concepts::base::Rotation TGroup>
void runBaseRotationTests() {
  // TODO
}

template <concepts::accessors::TxTy TGroup>
void runTxTyTests() {
  {
    auto g = TGroup::fromTx(0.7);
    SOPHUS_ASSERT_NEAR(
        g.translation().x(), 0.7, kEpsilon<typename TGroup::Scalar>);
    SOPHUS_ASSERT_NEAR(
        g.translation().y(),
        0.0,
        kEpsilon<typename TGroup::Scalar>,
        "{}",
        g.translation());
  }
  {
    auto g = TGroup::fromTy(-0.9);
    SOPHUS_ASSERT_NEAR(
        g.translation().y(),
        -0.9,
        kEpsilon<typename TGroup::Scalar>,
        "{}",
        g.translation());
    SOPHUS_ASSERT_NEAR(
        g.translation().x(),
        0.0,
        kEpsilon<typename TGroup::Scalar>,
        "{}",
        g.translation());
  }
}

template <concepts::accessors::TxTyTz TGroup>
void runTxTyTzTests() {
  {
    auto g = TGroup::fromTx(0.7);
    SOPHUS_ASSERT_NEAR(
        g.translation().x(), 0.7, kEpsilon<typename TGroup::Scalar>);
    SOPHUS_ASSERT_LE(
        g.translation().template tail<2>().norm(),
        kEpsilon<typename TGroup::Scalar>);
  }
  {
    auto g = TGroup::fromTy(-0.9);
    SOPHUS_ASSERT_NEAR(
        g.translation().y(),
        -0.9,
        kEpsilon<typename TGroup::Scalar>,
        "{}",
        g.translation());
    SOPHUS_ASSERT_NEAR(
        g.translation().x(),
        0.0,
        kEpsilon<typename TGroup::Scalar>,
        "{}",
        g.translation());
    SOPHUS_ASSERT_NEAR(
        g.translation().z(),
        0.0,
        kEpsilon<typename TGroup::Scalar>,
        "{}",
        g.translation());
  }
  {
    auto g = TGroup::fromTz(0.9);
    SOPHUS_ASSERT_NEAR(
        g.translation().z(),
        0.9,
        kEpsilon<typename TGroup::Scalar>,
        "{}",
        g.translation());
    SOPHUS_ASSERT_LE(
        g.translation().template head<2>().norm(),
        kEpsilon<typename TGroup::Scalar>);
  }
}

template <concepts::accessors::Isometry2 TGroup>
void runIsometry2AccessorTests() {
  runIsometryAccessorTests<TGroup>();
  runRotation2AccessorTests<TGroup>();
  runTxTyTests<TGroup>();
}

template <concepts::accessors::Isometry3 TGroup>
void runIsometry3AccessorTests() {
  runIsometryAccessorTests<TGroup>();
  runRotation3AccessorTests<TGroup>();
  runTxTyTzTests<TGroup>();
}

template <concepts::accessors::Similarity2 TGroup>
void runSimilarity2AccessorTests() {
  runSimilarityAccessorTests<TGroup>();
  runSpiralSimilarity2AccessorTests<TGroup>();
  runTxTyTests<TGroup>();
}

template <concepts::accessors::Similarity3 TGroup>
void runSimilarity3AccessorTests() {
  runSimilarityAccessorTests<TGroup>();
  runSpiralSimilarity3AccessorTests<TGroup>();
  runTxTyTzTests<TGroup>();
}

///

template <concepts::Rotation2 TGroup>
void runRotation2UnitTests() {
  runRotation2AccessorTests<TGroup>();
  runBaseRotationTests<TGroup>();
  runUnitComplexTests<TGroup>();
}

template <concepts::Rotation3 TGroup>
void runRotation3UnitTests() {
  runRotation3AccessorTests<TGroup>();
  runBaseRotationTests<TGroup>();
  runUnitQuaternionTests<TGroup>();
}

template <concepts::Isometry2 TGroup>
void runIsometry2UnitTests() {
  runIsometry2AccessorTests<TGroup>();
  runUnitComplexTests<TGroup>();
}

template <concepts::Isometry3 TGroup>
void runIsometry3UnitTests() {
  runIsometry3AccessorTests<TGroup>();
  runUnitQuaternionTests<TGroup>();
}

template <concepts::SpiralSimilarity2 TGroup>
void runSpiralSimilarity2UnitTests() {
  runSpiralSimilarity2AccessorTests<TGroup>();
}

template <concepts::SpiralSimilarity3 TGroup>
void runSpiralSimilarity3UnitTests() {
  runSpiralSimilarity3AccessorTests<TGroup>();
}

template <concepts::Similarity2 TGroup>
void runSimilarity2UnitTests() {
  runSimilarity2AccessorTests<TGroup>();
  runSpiralSimilarity2AccessorTests<TGroup>();
}

template <concepts::Similarity3 TGroup>
void runSimilarity3UnitTests() {
  runSimilarity3AccessorTests<TGroup>();
  runSpiralSimilarity3AccessorTests<TGroup>();
}
}  // namespace test
}  // namespace sophus
