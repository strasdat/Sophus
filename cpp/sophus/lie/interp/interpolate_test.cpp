// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/interp/interpolate.h"

#include "sophus/lie/interp/average.h"
#include "sophus/lie/scaling_translation.h"

#include <gtest/gtest.h>

namespace sophus::test {

template <
    template <class>
    class TGenericGroup,
    class TScalar,
    concepts::LieGroup TGroup = TGenericGroup<TScalar>>
struct InterpolatePropTestSuite {
  using Group = TGroup;

  using Scalar = typename Group::Scalar;

  static int constexpr kDof = Group::kDof;
  static int constexpr kNumParams = Group::kNumParams;
  static int constexpr kPointDim = Group::kPointDim;
  static int constexpr kAmbientDim = Group::kAmbientDim;

  static decltype(Group::paramsExamples()) const kParamsExamples;
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
    // TODO: Improve accuracy of ``interpolate`` (and hence ``exp`` and `log``)
    //       so that we can use more accurate bounds in these tests,i.e.
    //       ``eps`` instead of ``sqrt_eps``.

    for (size_t params_id1 = 0; params_id1 < kParamsExamples.size();
         ++params_id1) {
      Params params1 = SOPHUS_AT(kParamsExamples, params_id1);
      Group foo_from_bar = Group::fromParams(params1);
      for (size_t params_id2 = 0; params_id2 < kParamsExamples.size();
           ++params_id2) {
        Params params2 = SOPHUS_AT(kParamsExamples, params_id2);
        Group foo_from_daz = Group::fromParams(params2);

        // Test boundary conditions ``alpha=0`` and ``alpha=1``.
        Group foo_t_quiz = interpolate(foo_from_bar, foo_from_daz, Scalar(0));
        SOPHUS_ASSERT_NEAR(
            foo_t_quiz.matrix(), foo_from_bar.matrix(), sqrt_eps, "");
        foo_t_quiz = interpolate(foo_from_bar, foo_from_daz, Scalar(1));
        SOPHUS_ASSERT_NEAR(
            foo_t_quiz.matrix(), foo_from_daz.matrix(), 10.0 * sqrt_eps, "");
      }
    }
    for (Scalar alpha :
         {Scalar(0.1), Scalar(0.5), Scalar(0.75), Scalar(0.99)}) {
      for (size_t params_id1 = 0; params_id1 < kParamsExamples.size();
           ++params_id1) {
        Params params1 = SOPHUS_AT(kParamsExamples, params_id1);
        Group foo_from_bar = Group::fromParams(params1);
        for (size_t params_id2 = 0; params_id2 < kParamsExamples.size();
             ++params_id2) {
          Params params2 = SOPHUS_AT(kParamsExamples, params_id2);
          Group foo_from_daz = Group::fromParams(params2);
          Group foo_t_quiz = interpolate(foo_from_bar, foo_from_daz, alpha);
          // test left-invariance:
          //
          // dash_T_foo * interp(foo_from_bar, foo_from_daz)
          // == interp(dash_T_foo * foo_from_bar, dash_T_foo *
          // foo_from_daz)

          if ((foo_from_bar.inverse() * foo_from_daz)
                  .hasShortestPathAmbiguity()) {
            // skip check since there is a shortest path ambiguity
            continue;
          }
          for (size_t params_id3 = 0; params_id3 < kParamsExamples.size();
               ++params_id3) {
            Params params3 = SOPHUS_AT(kParamsExamples, params_id3);
            Group dash_from_foo = Group::fromParams(params3);
            Group dash_t_quiz = interpolate(
                dash_from_foo * foo_from_bar,
                dash_from_foo * foo_from_daz,
                alpha);
            SOPHUS_ASSERT_NEAR(
                dash_t_quiz.matrix(),
                (dash_from_foo * foo_t_quiz).matrix(),
                500 * sqrt_eps,
                "{}",
                group_name);
          }
          // test inverse-invariance:
          //
          // interp(foo_from_bar, foo_from_daz).inverse()
          // == interp(foo_from_bar.inverse(),dash_T_foo.inverse())
          Group quiz_t_foo = interpolate(
              foo_from_bar.inverse(), foo_from_daz.inverse(), alpha);
          SOPHUS_ASSERT_NEAR(
              quiz_t_foo.inverse().matrix(),
              foo_t_quiz.matrix(),
              500.0 * sqrt_eps,
              "");
        }
      }
      for (size_t params_id1 = 0; params_id1 < kParamsExamples.size();
           ++params_id1) {
        Params params1 = SOPHUS_AT(kParamsExamples, params_id1);
        Group bar_from_foo = Group::fromParams(params1);
        for (size_t params_id2 = 0; params_id2 < kParamsExamples.size();
             ++params_id2) {
          Params params2 = SOPHUS_AT(kParamsExamples, params_id2);
          Group baz_from_foo = Group::fromParams(params2);
          Group quiz_t_foo = interpolate(bar_from_foo, baz_from_foo, alpha);
          // test right-invariance:
          //
          // interp(bar_from_foo, bar_from_foo) * foo_T_dash
          // == interp(bar_from_foo * foo_T_dash, bar_from_foo*
          // foo_T_dash)
          if ((bar_from_foo * baz_from_foo.inverse())
                  .hasShortestPathAmbiguity()) {
            // skip check since there is a shortest path ambiguity
            continue;
          }
          for (size_t params_id3 = 0; params_id3 < kParamsExamples.size();
               ++params_id3) {
            Params params3 = SOPHUS_AT(kParamsExamples, params_id3);
            Group foo_from_dash = Group::fromParams(params3);
            Group quiz_t_dash = interpolate(
                bar_from_foo * foo_from_dash,
                baz_from_foo * foo_from_dash,
                alpha);
            SOPHUS_ASSERT_NEAR(
                quiz_t_dash.matrix(),
                (quiz_t_foo * foo_from_dash).matrix(),
                500 * sqrt_eps,
                "");
          }
        }
      }
    }
    for (size_t params_id1 = 0; params_id1 < kParamsExamples.size();
         ++params_id1) {
      Params params1 = SOPHUS_AT(kParamsExamples, params_id1);
      Group foo_from_bar = Group::fromParams(params1);
      for (size_t params_id2 = 0; params_id2 < kParamsExamples.size();
           ++params_id2) {
        Params params2 = SOPHUS_AT(kParamsExamples, params_id2);
        Group foo_from_daz = Group::fromParams(params2);
        if ((foo_from_bar.inverse() * foo_from_daz)
                .hasShortestPathAmbiguity()) {
          // skip check since there is a shortest path ambiguity
          continue;
        }
        // test average({A, B}) == interp(A, B):
        Group foo_t_quiz = interpolate(foo_from_bar, foo_from_daz, 0.5);
        std::optional<Group> foo_t_iaverage = iterativeMean(
            std::array<Group, 2>({{foo_from_bar, foo_from_daz}}), 20);
        std::optional<Group> foo_t_average =
            average(std::array<Group, 2>({{foo_from_bar, foo_from_daz}}));
        SOPHUS_ASSERT(
            bool(foo_t_average),
            "log(foo_from_bar): %\nlog(foo_from_daz): %",
            transpose(foo_from_bar.log()),
            transpose(foo_from_daz.log()),
            "");
        if (foo_t_average) {
          SOPHUS_ASSERT_NEAR(
              foo_t_quiz.matrix(),
              foo_t_average->matrix(),
              100.0 * sqrt_eps,
              "log(foo_from_bar): %\nlog(foo_from_daz): %\n"
              "log(interp): %\nlog(average): %",
              transpose(foo_from_bar.log()),
              transpose(foo_from_daz.log()),
              transpose(foo_t_quiz.log()),
              transpose(foo_t_average->log()),
              "");
        }
        SOPHUS_ASSERT(
            bool(foo_t_iaverage),
            "log(foo_from_bar): %\nlog(foo_from_daz): %\n"
            "log(interp): %\nlog(iaverage): %",
            transpose(foo_from_bar.log()),
            transpose(foo_from_daz.log()),
            transpose(foo_t_quiz.log()),
            transpose(foo_t_iaverage->log()),
            "");
        if (foo_t_iaverage) {
          SOPHUS_ASSERT_NEAR(
              foo_t_quiz.matrix(),
              foo_t_iaverage->matrix(),
              100 * sqrt_eps,
              "log(foo_from_bar): %\nlog(foo_from_daz): %",
              transpose(foo_from_bar.log()),
              transpose(foo_from_daz.log()),
              "");
        }
      }
    }
  }
};

template <
    template <class>
    class TGenericGroup,
    class TScalar,
    concepts::LieGroup TGroup>
decltype(TGroup::paramsExamples()) const
    InterpolatePropTestSuite<TGenericGroup, TScalar, TGroup>::kParamsExamples =
        TGroup::paramsExamples();

template <
    template <class>
    class TGenericGroup,
    class TScalar,
    concepts::LieGroup TGroup>
decltype(TGroup::tangentExamples()) const
    InterpolatePropTestSuite<TGenericGroup, TScalar, TGroup>::kTangentExamples =
        TGroup::tangentExamples();

template <
    template <class>
    class TGenericGroup,
    class TScalar,
    concepts::LieGroup TGroup>
decltype(pointExamples<typename TGroup::Scalar, TGroup::kPointDim>()) const
    InterpolatePropTestSuite<TGenericGroup, TScalar, TGroup>::kPointExamples =
        pointExamples<Scalar, kPointDim>();

TEST(lie_groups, linterpolate_prop_tests) {
  InterpolatePropTestSuite<Translation2, double>::runAllTests("Translation2");
  InterpolatePropTestSuite<Translation3, double>::runAllTests("Translation3");

  InterpolatePropTestSuite<Rotation2, double>::runAllTests("Rotation2");
  InterpolatePropTestSuite<Rotation3, double>::runAllTests("Rotation3");
  InterpolatePropTestSuite<Isometry2, double>::runAllTests("Isometry2");
  InterpolatePropTestSuite<Isometry3, double>::runAllTests("Isometry3");

  InterpolatePropTestSuite<SpiralSimilarity2, double>::runAllTests(
      "SpiralSimilarity2");
  InterpolatePropTestSuite<SpiralSimilarity3, double>::runAllTests(
      "SpiralSimilarity3");
  InterpolatePropTestSuite<Similarity2, double>::runAllTests("Similarity2");
  InterpolatePropTestSuite<Similarity3, double>::runAllTests("Similarity3");

  InterpolatePropTestSuite<Scaling2, double>::runAllTests("Scaling2");
  InterpolatePropTestSuite<Scaling3, double>::runAllTests("Scaling3");
  InterpolatePropTestSuite<ScalingTranslation2, double>::runAllTests(
      "ScalingTranslation2");
  InterpolatePropTestSuite<ScalingTranslation3, double>::runAllTests(
      "ScalingTranslation3");
}
}  // namespace sophus::test
