// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/interp/spline.h"

#include "sophus/calculus/num_diff.h"
#include "sophus/lie/isometry2.h"
#include "sophus/lie/isometry3.h"
#include "sophus/lie/rotation2.h"
#include "sophus/lie/rotation3.h"
#include "sophus/lie/scaling.h"
#include "sophus/lie/scaling_translation.h"
#include "sophus/lie/similarity2.h"
#include "sophus/lie/similarity3.h"
#include "sophus/lie/spiral_similarity2.h"
#include "sophus/lie/spiral_similarity3.h"
#include "sophus/lie/translation.h"

#include <gtest/gtest.h>

namespace sophus::test {

template <concepts::LieGroup TGroup>
struct SplinePropTestSuite {
  using Group = TGroup;

  using Scalar = typename Group::Scalar;

  static int constexpr kDof = Group::kDof;
  static int constexpr kNumParams = Group::kNumParams;
  static int constexpr kPointDim = Group::kPointDim;
  static int constexpr kAmbientDim = Group::kAmbientDim;

  static decltype(Group::elementExamples()) const kElementExamples;
  static decltype(Group::tangentExamples()) const kTangentExamples;
  static decltype(pointExamples<Scalar, Group::kPointDim>())
      const kPointExamples;

  using Tangent = Eigen::Vector<Scalar, kDof>;
  using Params = Eigen::Vector<Scalar, kNumParams>;
  using Point = Eigen::Vector<Scalar, kPointDim>;
  using Matrix = Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim>;
  using CompactMatrix = Eigen::Matrix<Scalar, kPointDim, kAmbientDim>;

  static void runAllTests(std::string group_name) {
    using std::sqrt;
    Scalar const eps = kEpsilon<Scalar>;
    Scalar const sqrt_eps = sqrt(eps);

    for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
      Group world_from_foo = SOPHUS_AT(kElementExamples, g_id);
      for (size_t g_id2 = 0; g_id2 < kElementExamples.size(); ++g_id2) {
        Group world_from_bar = SOPHUS_AT(kElementExamples, g_id2);
        std::vector<Group> control_poses;
        control_poses.push_back(
            interpolate(world_from_foo, world_from_bar, 0.0));

        for (double p = 0.2; p < 1.0; p += 0.2) {
          Group t_world_inter = interpolate(world_from_foo, world_from_bar, p);
          control_poses.push_back(t_world_inter);
        }

        BasisSplineImpl<Group> spline(control_poses, 1.0);

        Group t = spline.parentFromSpline(0.0, 1.0);
        Group t2 = spline.parentFromSpline(1.0, 0.0);

        SOPHUS_ASSERT_NEAR(
            t.matrix(), t2.matrix(), 10 * sqrt_eps, "parent_T_spline");

        Matrix dt_parent_from_spline = spline.dtParentFromSpline(0.0, 0.5);
        Matrix dt_parent_from_spline2 = curveNumDiff(
            [&](double u_bar) -> Matrix {
              return spline.parentFromSpline(0.0, u_bar).matrix();
            },
            0.5);
        SOPHUS_ASSERT_NEAR(
            dt_parent_from_spline,
            dt_parent_from_spline2,
            100 * sqrt_eps,
            "Dt_parent_T_spline: {}",
            group_name);
        Matrix dt2_parent_from_spline = spline.dt2ParentFromSpline(0.0, 0.5);
        Matrix dt2_parent_from_spline2 = curveNumDiff(
            [&](double u_bar) -> Matrix {
              return spline.dtParentFromSpline(0.0, u_bar).matrix();
            },
            0.5);
        SOPHUS_ASSERT_NEAR(
            dt2_parent_from_spline,
            dt2_parent_from_spline2,
            20 * sqrt_eps,
            "Dt2_parent_T_spline");

        for (double frac : {0.01, 0.25, 0.5, 0.9, 0.99}) {
          double t0 = 1.0;
          double delta_t = 0.1;
          BasisSpline<Group> spline(control_poses, t0, delta_t);
          double t = t0 + frac * delta_t;

          Matrix dt_parent_from_spline = spline.dtParentFromSpline(t);
          Matrix dt_parent_from_spline2 = curveNumDiff(
              [&](double t_bar) -> Matrix {
                return spline.parentFromSpline(t_bar).matrix();
              },
              t);
          SOPHUS_ASSERT_NEAR(
              dt_parent_from_spline,
              dt_parent_from_spline2,
              80 * sqrt_eps,
              "Dt_parent_T_spline");

          Matrix dt2_parent_from_spline = spline.dt2ParentFromSpline(t);
          Matrix dt2_parent_from_spline2 = curveNumDiff(
              [&](double t_bar) -> Matrix {
                return spline.dtParentFromSpline(t_bar).matrix();
              },
              t);
          SOPHUS_ASSERT_NEAR(
              dt2_parent_from_spline,
              dt2_parent_from_spline2,
              20 * sqrt_eps,
              "Dt2_parent_T_spline");
        }
      }
    }
  }
};

template <concepts::LieGroup TGroup>
decltype(TGroup::elementExamples())
    const SplinePropTestSuite<TGroup>::kElementExamples =
        TGroup::elementExamples();

template <concepts::LieGroup TGroup>
decltype(TGroup::tangentExamples())
    const SplinePropTestSuite<TGroup>::kTangentExamples =
        TGroup::tangentExamples();

template <concepts::LieGroup TGroup>
decltype(pointExamples<typename TGroup::Scalar, TGroup::kPointDim>())
    const SplinePropTestSuite<TGroup>::kPointExamples =
        pointExamples<Scalar, kPointDim>();

TEST(lie_groups, linterpolate_prop_tests) {
  SplinePropTestSuite<Scaling2<double>>::runAllTests("Scaling2");
  SplinePropTestSuite<Scaling3<double>>::runAllTests("Scaling3");

  SplinePropTestSuite<Translation2<double>>::runAllTests("Translation2");
  SplinePropTestSuite<Translation3<double>>::runAllTests("Translation3");
  SplinePropTestSuite<ScalingTranslation2<double>>::runAllTests(
      "ScalingTranslation2");
  SplinePropTestSuite<ScalingTranslation3<double>>::runAllTests(
      "ScalingTranslation3");

  SplinePropTestSuite<Rotation2<double>>::runAllTests("Rotation2");
  SplinePropTestSuite<Rotation3<double>>::runAllTests("Rotation3");
  SplinePropTestSuite<Isometry2<double>>::runAllTests("Isometry2");
  SplinePropTestSuite<Isometry3<double>>::runAllTests("Isometry3");

  SplinePropTestSuite<SpiralSimilarity2<double>>::runAllTests(
      "SpiralSimilarity2");
  SplinePropTestSuite<SpiralSimilarity3<double>>::runAllTests(
      "SpiralSimilarity3");
  SplinePropTestSuite<Similarity2<double>>::runAllTests("Similarity2");
  SplinePropTestSuite<Similarity3<double>>::runAllTests("Similarity3");
}
}  // namespace sophus::test
