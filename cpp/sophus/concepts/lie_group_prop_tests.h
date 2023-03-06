// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/calculus/num_diff.h"
#include "sophus/concepts/lie_group.h"
#include "sophus/linalg/vector_space.h"

namespace sophus {
namespace test {

template <concepts::LieGroup TGroup>
struct LieGroupPropTestSuite {
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

  static auto preservabilityTests(std::string group_name) -> void {
    if (kElementExamples.size() == 0) {
      return;
    }

    if (Group::kIsOriginPreserving) {
      for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
        Group g = SOPHUS_AT(kElementExamples, g_id);
        Point o;
        o.setZero();
        SOPHUS_ASSERT_NEAR(g * o, o, kEpsilon<Scalar>);
      }
    } else {
      size_t num_preserves = 0;
      size_t num = 0;
      for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
        Group g = SOPHUS_AT(kElementExamples, g_id);
        Point o;
        o.setZero();
        if ((g * o).norm() < kEpsilon<Scalar>) {
          ++num_preserves;
        }
        ++num;
      }
      float percentage = float(num_preserves) / float(num);
      FARM_ASSERT_LE(percentage, 0.75);
    }

    if (Group::kIsAxisDirectionPreserving) {
      for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
        Group g = SOPHUS_AT(kElementExamples, g_id);
        for (int d = 0; d < kPointDim; ++d) {
          Point p;
          p.setZero();
          p[d] = 1.0;
          UnitVector<Scalar, kPointDim> e =
              UnitVector<Scalar, kPointDim>::fromUnitVector(p);
          SOPHUS_ASSERT_NEAR((g * e).params(), e.params(), kEpsilon<Scalar>);
        }
      }
    } else {
      size_t num_preserves = 0;
      size_t num = 0;
      for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
        Group g = SOPHUS_AT(kElementExamples, g_id);
        for (int d = 0; d < kPointDim; ++d) {
          Point p;
          p.setZero();
          p[d] = 1.0;
          UnitVector<Scalar, kPointDim> e =
              UnitVector<Scalar, kPointDim>::fromUnitVector(p);

          if (((g * e).params() - e.params()).norm() < kEpsilon<Scalar>) {
            ++num_preserves;
          }
          ++num;
        }
      }
      float percentage = float(num_preserves) / float(num);
      FARM_ASSERT_LE(percentage, 0.75);
    }

    if (Group::kIsDirectionVectorPreserving) {
      for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
        Group g = SOPHUS_AT(kElementExamples, g_id);
        for (size_t point_id = 0; point_id < kPointExamples.size();
             ++point_id) {
          auto p = SOPHUS_AT(kPointExamples, point_id);
          if (p.norm() < kEpsilon<Scalar>) {
            continue;
          }
          UnitVector<Scalar, kPointDim> d =
              UnitVector<Scalar, kPointDim>::fromVectorAndNormalize(p);
          SOPHUS_ASSERT_NEAR((g * d).params(), d.params(), kEpsilon<Scalar>);
        }
      }
    } else {
      size_t num_preserves = 0;
      size_t num = 0;
      for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
        Group g = SOPHUS_AT(kElementExamples, g_id);
        for (size_t point_id = 0; point_id < kPointExamples.size();
             ++point_id) {
          auto p = SOPHUS_AT(kPointExamples, point_id);
          if (p.norm() < kEpsilon<Scalar>) {
            continue;
          }
          UnitVector<Scalar, kPointDim> d =
              UnitVector<Scalar, kPointDim>::fromVectorAndNormalize(p);
          if (((g * d).params() - d.params()).norm() < kEpsilon<Scalar>) {
            ++num_preserves;
          }
          ++num;
        }
      }
      float percentage = float(num_preserves) / float(num);
      FARM_ASSERT_LE(percentage, 0.75);
    }
  }

  static auto expTests(std::string group_name) -> void {
    for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
      Group g = SOPHUS_AT(kElementExamples, g_id);
      auto matrix_before = g.compactMatrix();
      auto matrix_after = Group::exp(g.log()).compactMatrix();

      SOPHUS_ASSERT_NEAR(
          matrix_before,
          matrix_after,
          kEpsilonSqrt<Scalar>,
          "`exp(log(g)) == g` Test for {}\n"
          "params #{}",
          group_name,
          g_id);
    }
    for (size_t i = 0; i < kTangentExamples.size(); ++i) {
      Tangent tangent = SOPHUS_AT(kTangentExamples, i);

      Group exp_inverse = Group::exp(tangent).inverse();
      Group exp_neg_tangent = Group::exp(-tangent);

      SOPHUS_ASSERT_NEAR(
          exp_inverse.compactMatrix(),
          exp_neg_tangent.compactMatrix(),
          0.001,
          "`exp(-t) == inv(exp(t))` Test for {}\n"
          "Group::exp(tangent): \n{}\ntangent #{} {}",
          group_name,
          Group::exp(tangent).compactMatrix(),
          i,
          tangent);
    }
  }

  static auto adjointTests(std::string group_name) -> void {
    for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
      Group g = SOPHUS_AT(kElementExamples, g_id);
      Matrix mat = g.matrix();
      Eigen::Matrix<Scalar, Group::kDof, Group::kDof> ad = g.adj();
      for (size_t tangent_id = 0; tangent_id < kTangentExamples.size();
           ++tangent_id) {
        Tangent x = SOPHUS_AT(kTangentExamples, tangent_id);
        Tangent ad1 = ad * x;
        Tangent ad2 = Group::vee(mat * Group::hat(x) * g.inverse().matrix());
        SOPHUS_ASSERT_NEAR(
            ad1,
            ad2,
            10 * kEpsilonSqrt<Scalar>,
            "`ad * x == vee(g * hat(x) * inv(g))` Test for {}\n"
            "Adj: {}"
            "tangent # {} ({})\n"
            "params # {} ({}); matrix:\n"
            "{}",
            group_name,
            ad,
            tangent_id,
            x.transpose(),
            g_id,
            g.params().transpose(),
            g.compactMatrix());
      }
    }
  }

  static auto hatTests(std::string group_name) -> void {
    for (size_t i = 0; i < kTangentExamples.size(); ++i) {
      Tangent tangent = SOPHUS_AT(kTangentExamples, i);
      SOPHUS_ASSERT_NEAR(
          tangent,
          Group::vee(Group::hat(tangent)),
          kEpsilonSqrt<Scalar>,
          "`t = vee(hat(t))` Test for {}, tangent #{}: {}",
          group_name,
          i,
          tangent.transpose());
    }
  }

  static auto groupOperationTests(std::string group_name) -> void {
    for (size_t g_id1 = 0; g_id1 < kElementExamples.size(); ++g_id1) {
      Group g1 = SOPHUS_AT(kElementExamples, g_id1);
      for (size_t g_id2 = 0; g_id2 < kElementExamples.size(); ++g_id2) {
        Group g2 = SOPHUS_AT(kElementExamples, g_id2);
        for (size_t g_id3 = 0; g_id3 < kElementExamples.size(); ++g_id3) {
          Group g3 = SOPHUS_AT(kElementExamples, g_id3);

          Group left_hugging = (g1 * g2) * g3;
          Group right_hugging = g1 * (g2 * g3);
          SOPHUS_ASSERT_NEAR(
              left_hugging.compactMatrix(),
              right_hugging.compactMatrix(),
              10.0 * kEpsilonSqrt<Scalar>,
              "`(g1*g2)*g3 == g1*(g2*g3)` Test for {}, #{}/#{}/#{}",
              group_name,
              g_id1,
              g_id2,
              g_id3);
        }
      }
    }

    for (size_t g_id1 = 0; g_id1 < kElementExamples.size(); ++g_id1) {
      Group foo_from_bar_transform = SOPHUS_AT(kElementExamples, g_id1);
      for (size_t g_id2 = 0; g_id2 < kElementExamples.size(); ++g_id2) {
        Group bar_from_daz_transform = SOPHUS_AT(kElementExamples, g_id2);

        Group daz_from_foo_transform_1 =
            bar_from_daz_transform.inverse() * foo_from_bar_transform.inverse();
        Group daz_from_foo_transform_2 =
            (foo_from_bar_transform * bar_from_daz_transform).inverse();

        SOPHUS_ASSERT_NEAR(
            daz_from_foo_transform_1.compactMatrix(),
            daz_from_foo_transform_2.compactMatrix(),
            10 * kEpsilonSqrt<Scalar>,
            "`ing(g2) * inv(g1) == inv(g1 *g2)` Test for {}, #{}/#{}",
            group_name,
            g_id1,
            g_id2);
      }
    }
  }

  static auto groupActionTests(std::string group_name) -> void {
    for (size_t point_id = 0; point_id < kPointExamples.size(); ++point_id) {
      auto point_in = SOPHUS_AT(kPointExamples, point_id);
      for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
        Group g = SOPHUS_AT(kElementExamples, g_id);
        Point out_point_from_matrix =
            g.compactMatrix() * Group::toAmbient(point_in);
        Point out_point_from_action = g * point_in;

        SOPHUS_ASSERT_NEAR(
            out_point_from_matrix,
            out_point_from_action,
            kEpsilonSqrt<Scalar>,
            "`g.matrix() * point == g * point`: {}\n"
            "point # {} ({})\n"
            "params # {} ({}); matrix:\n"
            "{}",
            group_name,
            point_id,
            point_in.transpose(),
            g_id,
            g.params().transpose(),
            g.compactMatrix());

        Point in_point_through_inverse = g.inverse() * out_point_from_matrix;
        SOPHUS_ASSERT_NEAR(
            point_in,
            in_point_through_inverse,
            10.0 * kEpsilonSqrt<Scalar>,
            "`inv(g) * (g * point) == point` Test for {}\n"
            "point # {} ({})\n"
            "params # {} ({}); matrix:\n"
            "{}",
            group_name,
            point_id,
            point_in.transpose(),
            g_id,
            g.params().transpose(),
            g.compactMatrix());
      }
    }
  }

  static void expJacobiansTest(std::string group_name) {
    if constexpr (kDof == 0) {
      return;
    } else {
      // for (size_t j = 0; j < kTangentExamples.size(); ++j) {
      //   Tangent a = SOPHUS_AT(kTangentExamples, j);
      //   Eigen::Matrix<Scalar, kNumParams, kDof> d = Group::dxExpX(a);
      //   Eigen::Matrix<Scalar, kNumParams, kDof> j_num =
      //       vectorFieldNumDiff<Scalar, kNumParams, kDof>(
      //           [](Tangent const& x) -> Params {
      //             return Group::exp(x).params();
      //           },
      //           a);

      //   SOPHUS_ASSERT_NEAR(
      //       d,
      //       j_num,
      //       10.0 * kEpsilonSqrt<Scalar>,
      //       "`dx exp(x)` Test {}",
      //       group_name);
      // }

      Tangent o;
      o.setZero();
      Eigen::Matrix<Scalar, kNumParams, kDof> j = Group::dxExpXAt0();
      Eigen::Matrix<Scalar, kNumParams, kDof> j_num =
          vectorFieldNumDiff<Scalar, kNumParams, kDof>(
              [](Tangent const& x) -> Params {
                Params p = Group::exp(x).params();
                return p;
              },
              o);
      SOPHUS_ASSERT_NEAR(
          j,
          j_num,
          10.0 * kEpsilonSqrt<Scalar>,
          "`dx exp(x)|x=0` Test {}",
          group_name);

      for (size_t point_id = 0; point_id < kPointExamples.size(); ++point_id) {
        Point point_in = SOPHUS_AT(kPointExamples, point_id);
        Eigen::Matrix<Scalar, kPointDim, kDof> j =
            Group::dxExpXTimesPointAt0(point_in);
        Tangent o;
        o.setZero();
        Eigen::Matrix<Scalar, kPointDim, kDof> const j_num =
            vectorFieldNumDiff<Scalar, kPointDim, kDof>(
                [point_in](Tangent const& x) {
                  return Group::exp(x) * point_in;
                },
                o);

        SOPHUS_ASSERT_NEAR(
            j,
            j_num,
            10.0 * kEpsilonSqrt<Scalar>,
            "expJacobiansTest #1: {}",
            group_name);
      }

      for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
        Group g = SOPHUS_AT(kElementExamples, g_id);

        Tangent o;
        o.setZero();
        Eigen::Matrix<Scalar, kNumParams, kDof> j = g.dxThisMulExpXAt0();
        Eigen::Matrix<Scalar, kNumParams, kDof> j_num =
            vectorFieldNumDiff<Scalar, kNumParams, kDof>(
                [g](Tangent const& x) -> Params {
                  return (g * Group::exp(x)).params();
                },
                o);

        SOPHUS_ASSERT_NEAR(
            j,
            j_num,
            10.0 * kEpsilonSqrt<Scalar>,
            "expJacobiansTest #2: {}",
            group_name);
      }

      for (size_t g_id = 0; g_id < kElementExamples.size(); ++g_id) {
        Group g = SOPHUS_AT(kElementExamples, g_id);

        Eigen::Matrix<Scalar, kDof, kDof> j =
            g.dxLogThisInvTimesXAtThis() * g.dxThisMulExpXAt0();
        Eigen::Matrix<Scalar, kDof, kDof> j_exp =
            Eigen::Matrix<Scalar, kDof, kDof>::Identity();

        SOPHUS_ASSERT_NEAR(
            j,
            j_exp,
            10.0 * kEpsilonSqrt<Scalar>,
            "expJacobiansTest #3: {}",
            group_name);
      }
    }
  }

  static auto runAllTests(std::string group_name) -> void {
    preservabilityTests(group_name);
    expTests(group_name);
    adjointTests(group_name);
    hatTests(group_name);

    groupOperationTests(group_name);
    groupActionTests(group_name);

    expJacobiansTest(group_name);
  }
};

template <concepts::LieGroup TGroup>
decltype(TGroup::elementExamples())
    const LieGroupPropTestSuite<TGroup>::kElementExamples =
        TGroup::elementExamples();

template <concepts::LieGroup TGroup>
decltype(TGroup::tangentExamples())
    const LieGroupPropTestSuite<TGroup>::kTangentExamples =
        TGroup::tangentExamples();

template <concepts::LieGroup TGroup>
decltype(pointExamples<typename TGroup::Scalar, TGroup::kPointDim>())
    const LieGroupPropTestSuite<TGroup>::kPointExamples =
        pointExamples<Scalar, kPointDim>();

// using namespace sophus;

// //   bool contructorAndAssignmentTest() {
// //     bool passed = true;
// //     for (Group foo_transform_bar : group_vec_) {
// //       Group foo_t2_bar = foo_transform_bar;
// //       SOPHUS_TEST_APPROX(
// //           passed,
// //           foo_transform_bar.matrix(),
// //           foo_t2_bar.matrix(),
// //           small_eps,
// //           "Copy constructor: %\nvs\n %",
// //           transpose(foo_transform_bar.matrix()),
// //           transpose(foo_t2_bar.matrix()));
// //       Group foo_t3_bar;
// //       foo_t3_bar = foo_transform_bar;
// //       SOPHUS_TEST_APPROX(
// //           passed,
// //           foo_transform_bar.matrix(),
// //           foo_t3_bar.matrix(),
// //           small_eps,
// //           "Copy assignment: %\nvs\n %",
// //           transpose(foo_transform_bar.matrix()),
// //           transpose(foo_t3_bar.matrix()));

// //       Group foo_t4_bar(foo_transform_bar.matrix());
// //       SOPHUS_TEST_APPROX(
// //           passed,
// //           foo_transform_bar.matrix(),
// //           foo_t4_bar.matrix(),
// //           small_eps,
// //           "Constructor from homogeneous matrix: %\nvs\n %",
// //           transpose(foo_transform_bar.matrix()),
// //           transpose(foo_t4_bar.matrix()));

// //       Eigen::Map<Group> foo_tmap_bar(foo_transform_bar.data());
// //       Group foo_t5_bar = foo_tmap_bar;
// //       SOPHUS_TEST_APPROX(
// //           passed,
// //           foo_transform_bar.matrix(),
// //           foo_t5_bar.matrix(),
// //           small_eps,
// //           "Assignment from Eigen::Map type: %\nvs\n %",
// //           transpose(foo_transform_bar.matrix()),
// //           transpose(foo_t5_bar.matrix()));

// //       Eigen::Map<Group const> foo_tcmap_bar(foo_transform_bar.data());
// //       Group foo_t6_bar;
// //       foo_t6_bar = foo_tcmap_bar;
// //       SOPHUS_TEST_APPROX(
// //           passed,
// //           foo_transform_bar.matrix(),
// //           foo_t5_bar.matrix(),
// //           small_eps,
// //           "Assignment from Eigen::Map type: %\nvs\n %",
// //           transpose(foo_transform_bar.matrix()),
// //           transpose(foo_t5_bar.matrix()));

// //       Group i;
// //       Eigen::Map<Group> foo_tmap2_bar(i.data());
// //       foo_tmap2_bar = foo_transform_bar;
// //       SOPHUS_TEST_APPROX(
// //           passed,
// //           foo_tmap2_bar.matrix(),
// //           foo_transform_bar.matrix(),
// //           small_eps,
// //           "Assignment to Eigen::Map type: %\nvs\n %",
// //           transpose(foo_tmap2_bar.matrix()),
// //           transpose(foo_transform_bar.matrix()));
// //     }
// //     return passed;
// //   }

// //   bool derivativeTest() {
// //     bool passed = true;

// //     Group g;
// //     for (int i = 0; i < kDof; ++i) {
// //       Transformation gi = g.dxiExpmatXAt0(i);
// //       Transformation gi2 = curveNumDiff(
// //           [i](Scalar xi) -> Transformation {
// //             Tangent x;
// //             setToZero(x);
// //             setElementAt(x, xi, i);
// //             return Group::exp(x).matrix();
// //           },
// //           Scalar(0));
// //       SOPHUS_TEST_APPROX(
// //           passed, gi, gi2, small_eps_sqrt, "Dxi_exp_x_matrix_at_ case
// %", i);
// //     }

// //     return passed;
// //   }

// // template <class G>
// // void productTest(std::string group_name) {
// //   bool passed = true;

// //   for (size_t params_id = 1; params_id < G::paramsExamples().size();
// //        ++params_id) {
// //     auto params1 = G::paramsExamples()[params_id - 1];
// //     auto params2 = G::paramsExamples()[params_id];
// //     auto g1 = G::fromParams(params);
// //     auto g2 = G::fromParams(params);

// //     G product = g1 * g2;
// //     g1 *= g2;
// //     SOPHUS_ASSERT_NEAR(
// //         g1.matrix(), product.matrix(), small_eps, "Product case: %", i);
// //   }
// //   return passed;
// // }

// // bool expMapTest() {
// //   bool passed = true;
// //   for (size_t i = 0; i < tangent_vec_.size(); ++i) {
// //     Tangent omega = tangent_vec_[i];
// //     Transformation exp_x = Group::exp(omega).matrix();
// //     Transformation expmap_hat_x = (Group::hat(omega)).exp();
// //     SOPHUS_TEST_APPROX(
// //         passed,
// //         exp_x,
// //         expmap_hat_x,
// //         Scalar(10) * small_eps,
// //         "expmap(hat(x)) - exp(x) case: %",
// //         i);
// //   }
// //   return passed;
// // }

// // template <class G>
// // bool lieBracketTest(std::string group_name) {
// //   for (size_t i = 0; i < tangent_vec_.size(); ++i) {
// //     for (size_t j = 0; j < tangent_vec_.size(); ++j) {
// //       Tangent tangent1 = Group::lieBracket(tangent_vec_[i],
// //       tangent_vec_[j]); Transformation hati =
// Group::hat(tangent_vec_[i]);
// //       Transformation hatj = Group::hat(tangent_vec_[j]);

// //       Tangent tangent2 = Group::vee(hati * hatj - hatj * hati);
// //       SOPHUS_ASSERT_NEAR(
// //           passed,
// //           tangent1,
// //           tangent2,
// //           small_eps,
// //           "lieBracketTest {}",
// //           group_name);
// //     }
// //   }
// // }

// //   template <class TS = Scalar>
// //   std::enable_if_t<std::is_same_v<TS, float>::value, bool> testSpline()
// {
// //     // skip tests for Scalar == float
// //     return true;
// //   }

// //   template <class TS = Scalar>
// //   std::enable_if_t<!std::is_same_v<TS, float>::value, bool> testSpline()
// {
// //     // run tests for Scalar != float
// //     bool passed = true;

// //     for (Group const& t_world_foo : group_vec_) {
// //       for (Group const& t_world_bar : group_vec_) {
// //         std::vector<Group> control_poses;
// //         control_poses.push_back(interpolate(t_world_foo, t_world_bar,
// 0.0));

// //         for (double p = 0.2; p < 1.0; p += 0.2) {
// //           Group t_world_inter = interpolate(t_world_foo, t_world_bar,
// p);
// //           control_poses.push_back(t_world_inter);
// //         }

// //         BasisSplineImpl<Group> spline(control_poses, 1.0);

// //         Group t = spline.parentFromSpline(0.0, 1.0);
// //         Group t2 = spline.parentFromSpline(1.0, 0.0);

// //         SOPHUS_TEST_APPROX(
// //             passed,
// //             t.matrix(),
// //             t2.matrix(),
// //             10 * small_eps_sqrt,
// //             "parent_T_spline");

// //         Transformation dt_parent_t_spline =
// spline.dtParentFromSpline(0.0,
// //         0.5); Transformation dt_parent_t_spline2 = curveNumDiff(
// //             [&](double u_bar) -> Transformation {
// //               return spline.parentFromSpline(0.0, u_bar).matrix();
// //             },
// //             0.5);
// //         SOPHUS_TEST_APPROX(
// //             passed,
// //             dt_parent_t_spline,
// //             dt_parent_t_spline2,
// //             100 * small_eps_sqrt,
// //             "Dt_parent_T_spline");
// //         Transformation dt2_parent_t_spline =
// //             spline.dt2ParentFromSpline(0.0, 0.5);
// //         Transformation dt2_parent_t_spline2 = curveNumDiff(
// //             [&](double u_bar) -> Transformation {
// //               return spline.dtParentFromSpline(0.0, u_bar).matrix();
// //             },
// //             0.5);
// //         SOPHUS_TEST_APPROX(
// //             passed,
// //             dt2_parent_t_spline,
// //             dt2_parent_t_spline2,
// //             20 * small_eps_sqrt,
// //             "Dt2_parent_T_spline");

// //         for (double frac : {0.01, 0.25, 0.5, 0.9, 0.99}) {
// //           double t0 = 1.0;
// //           double delta_t = 0.1;
// //           BasisSpline<Group> spline(control_poses, t0, delta_t);
// //           double t = t0 + frac * delta_t;

// //           Transformation dt_parent_t_spline =
// spline.dtParentFromSpline(t);
// //           Transformation dt_parent_t_spline2 = curveNumDiff(
// //               [&](double t_bar) -> Transformation {
// //                 return spline.parentFromSpline(t_bar).matrix();
// //               },
// //               t);
// //           SOPHUS_TEST_APPROX(
// //               passed,kDof
// //               dt_parent_t_spline,
// //               dt_parent_t_spline2,
// //               80 * small_eps_sqrt,
// //               "Dt_parent_T_spline");

// //           Transformation dt2_parent_t_spline =
// spline.dt2ParentFromSpline(t);
// //           Transformation dt2_parent_t_spline2 = curveNumDiff(
// //               [&](double t_bar) -> Transformation {
// //                 return spline.dtParentFromSpline(t_bar).matrix();
// //               },
// //               t);
// //           SOPHUS_TEST_APPROX(
// //               passed,
// //               dt2_parent_t_spline,
// //               dt2_parent_t_spline2,
// //               20 * small_eps_sqrt,
// //               "Dt2_parent_T_spline");
// //         }
// //       }
// //     }
// //     return passed;
// //   }
}  // namespace test
}  // namespace sophus
