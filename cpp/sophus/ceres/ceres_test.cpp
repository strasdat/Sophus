// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/ceres/manifold.h"
#include "sophus/common/common.h"
#include "sophus/lie/isometry2.h"
#include "sophus/lie/isometry3.h"
#include "sophus/lie/similarity2.h"
#include "sophus/lie/similarity3.h"
#include "sophus/linalg/vector_space.h"

#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <iostream>

namespace sophus::test {

template <class TScalar>
struct Random {
  template <class TRt>
  static TScalar sample(TRt& rng) {
    std::normal_distribution<double> rnorm;
    static_assert(
        TScalar::RowsAtCompileTime >= 0,
        "Matrix should have known size at compile-time");
    static_assert(
        TScalar::ColsAtCompileTime >= 0,
        "Matrix should have known size at compile-time");
    TScalar res;
    for (Eigen::Index i = 0; i < res.size(); ++i) {
      res.data()[i] = rnorm(rng);
    }
    return res;
  }
};

template <template <class> class TLieGroup>
struct LieGroupCeresTests {
  template <class TScalar>
  using LieGroup = TLieGroup<TScalar>;
  using LieGroupF64 = LieGroup<double>;

  static int constexpr kDof = LieGroupF64::kDof;
  static int constexpr kNumParams = LieGroupF64::kNumParams;
  static int constexpr kPointDim = LieGroupF64::kPointDim;

  using PointF64 = Eigen::Vector<double, kPointDim>;
  using TangentF64 = Eigen::Vector<double, kDof>;

  struct TestLieGroupCostFunctor {
    TestLieGroupCostFunctor(LieGroupF64 const& foo_from_world_transform)
        : foo_from_world_transform(foo_from_world_transform) {}

    template <class TScalar>
    bool operator()(
        TScalar const* const raw_world_from_foo, TScalar* raw_residuals) const {
      Eigen::Map<Eigen::Vector<TScalar, kDof>> residuals(raw_residuals);

      LieGroup<TScalar> world_from_foo_transform =
          LieGroup<TScalar>::fromParams(
              Eigen::Map<Eigen::Vector<TScalar, kNumParams> const>(
                  raw_world_from_foo));

      // // We are able to mix Sophus types with doubles and Jet types withou
      // // needing to cast to TScalar.
      // residuals = (foo_from_world_transform *
      // world_from_foo_transform).log();
      // // Reverse order of multiplication. This forces the compiler to verify
      // // that (Jet, double) and (double, Jet) LieGroup multiplication work
      // // correctly.
      // residuals = (world_from_foo_transform *
      // foo_from_world_transform).log(); Finally, ensure that Jet-to-Jet
      // multiplication works.
      residuals = (world_from_foo_transform *
                   foo_from_world_transform.template cast<TScalar>())
                      .log();
      return true;
    }

    LieGroupF64 foo_from_world_transform;
  };
  struct TestPointCostFunctor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TestPointCostFunctor(PointF64 const& point_in_foo)
        : point_in_foo(point_in_foo) {}

    template <class TScalar>
    bool operator()(
        TScalar const* const raw_world_from_foo,
        TScalar const* const raw_point_in_world,
        TScalar* raw_residuals) const {
      LieGroup<TScalar> world_from_foo_transform =
          LieGroup<TScalar>::fromParams(
              Eigen::Map<Eigen::Vector<TScalar, kNumParams> const>(
                  raw_world_from_foo));
      Eigen::Map<Eigen::Vector<TScalar, kPointDim> const> point_in_world(
          raw_point_in_world);
      Eigen::Map<Eigen::Vector<TScalar, kPointDim>> residuals(raw_residuals);

      Eigen::Vector<TScalar, kPointDim> point_in_foo_prime =
          world_from_foo_transform.inverse() * point_in_world;

      residuals = point_in_foo_prime - point_in_foo;
      return true;
    }

    LieGroupF64 foo_from_world_transform;
    PointF64 point_in_foo;
  };

  struct TestGraphFunctor {
    template <class TScalar>
    bool operator()(
        TScalar const* raw_world_from_foo,
        TScalar const* raw_world_from_bar,
        TScalar* raw_residuals) const {
      using LieGroup = TLieGroup<TScalar>;
      Eigen::Map<Eigen::Vector<TScalar, kDof>> residuals(raw_residuals);

      Eigen::Map<LieGroup const> world_from_foo_transform(raw_world_from_foo);
      Eigen::Map<LieGroup const> world_from_bar_transform(raw_world_from_bar);

      residuals =
          (bar_from_foo_transform *
           (world_from_bar_transform.inverse() * world_from_foo_transform))
              .log();
      return true;
    }

    TestGraphFunctor(LieGroupF64 const& bar_from_foo_transform)
        : bar_from_foo_transform(bar_from_foo_transform) {}
    const LieGroupF64 bar_from_foo_transform;
  };

  void testAll() {
    // for (size_t i = 0; i < group_vec.size(); ++i) {
    //   for (size_t j = 0; j < group_vec.size(); ++j) {
    //     if (i == j) {
    //       continue;
    //     }
    //     for (size_t k = 0; k < point_vec.size(); ++k) {
    //       for (size_t l = 0; l < point_vec.size(); ++l) {
    //         if (k == l) {
    //           continue;
    //         }
    //         std::cerr << "Simple test #" << i << ", " << j << ", " << k << ",
    //         "
    //                   << l;

    //         test(group_vec[i], group_vec[j], point_vec[k], point_vec[l]);
    //       }
    //     }
    //   }
    // }

    // int ns[] = {20, 40, 80, 160};
    // for (auto k_matrix_dim : ns) {
    //   std::cerr << "Averaging test: kMatrixDim = " << k_matrix_dim;
    //   testAveraging(k_matrix_dim, .5, .1);
    // }
  }

  // void testAveraging(
  //     const size_t num_vertices,
  //     double const sigma_init,
  //     double const sigma_observation) {
  //   if (num_vertices == 0u) {
  //     return;
  //   }
  //   double const sigma_init_elementwise = sigma_init / std::sqrt(kDof);
  //   double const sigma_observation_elementwise =
  //       sigma_observation / std::sqrt(kDof);
  //   // Running Lie group averaging on a K_n graph with a random
  //   initialization
  //   // noise and random noise in observations
  //   ::ceres::Problem problem;

  //   // "Random" initialization in order to keep tests repeatable
  //   std::mt19937 rng(2021);
  //   std::vector<LieGroupF64> vec(num_vertices);

  //   std::vector<LieGroupF64> v_estimate;
  //   v_estimate.reserve(num_vertices);
  //   double initial_error = 0.;
  //   auto parametrization = new sophus::ceres::Manifold<TLieGroup>;

  //   // All vertices are initialized with an i.i.d noise with normal
  //   // distribution; Scaling is adjusted in order to maintain the same
  //   // expectation of squared norm for all groups
  //   for (size_t i = 0; i < num_vertices; ++i) {
  //     auto& v = vec[i];
  //     v = LieGroupF64::sampleUniform(rng);
  //     const TangentF64 delta_log =
  //         Random<TangentF64>::sample(rng) * sigma_init_elementwise;
  //     const LieGroupF64 delta = LieGroupF64::exp(delta_log);
  //     v_estimate.emplace_back(v * delta);
  //     initial_error += squaredNorm(delta_log);
  //     problem.AddParameterBlock(
  //         v_estimate.back().data(), LieGroupF64::kNumParams,
  //         parametrization);
  //   }

  //   // For simplicity of graph generation, we use a complete (undirected)
  //   graph.
  //   // Each edge (observation) has i.i.d noise with multivariate normal
  //   // distribution; Scaling is adjusted in order to maintain the same
  //   // expectation of squared norm for all groups
  //   for (size_t i = 0; i < num_vertices; ++i) {
  //     for (size_t j = i + 1; j < num_vertices; ++j) {
  //       LieGroupF64 diff = vec[i].inverse() * vec[j];
  //       auto const delta_log =
  //           Random<typename LieGroupF64::Tangent>::sample(rng) *
  //           sigma_observation_elementwise;
  //       auto const delta = LieGroupF64::exp(delta_log);
  //       ::ceres::CostFunction* cost = new ::ceres::AutoDiffCostFunction<
  //           TestGraphFunctor,
  //           LieGroupF64::kDof,
  //           LieGroupF64::kNumParams,
  //           LieGroupF64::kNumParams>(new TestGraphFunctor(diff * delta));
  //       // For real-world problems you should consider using robust
  //       // loss-function
  //       problem.AddResidualBlock(
  //           cost, nullptr, v_estimate[i].data(), v_estimate[j].data());
  //     }
  //   }

  //   ::ceres::Solver::Options options;
  //   options.gradient_tolerance = 1e-2 * sophus::kEpsilonF64;
  //   options.function_tolerance = 1e-2 * sophus::kEpsilonF64;
  //   options.parameter_tolerance = 1e-2 * sophus::kEpsilonF64;
  //   options.linear_solver_type = ::ceres::SPARSE_NORMAL_CHOLESKY;

  //   ::ceres::Solver::Summary summary;
  //   Solve(options, &problem, &summary);

  //   // Computing final error in the estimates
  //   double final_error = 0;
  //   for (int i = 0; i < num_vertices; ++i) {
  //     final_error += squaredNorm((vec[i].inverse() * v_estimate[i]).log());
  //   }

  //   // Expecting reasonable decrease of both estimates' errors and residuals
  //   SOPHUS_ASSERT(summary.final_cost < .25 * summary.initial_cost);
  //   SOPHUS_ASSERT(final_error < .25 * initial_error);
  // }

  // bool test(
  //     LieGroupF64 const& t_w_targ,
  //     LieGroupF64 const& t_w_init,
  //     PointF64 const& point_a_init,
  //     PointF64 const& point_b) {
  //   static int constexpr kNumPointParameters = PointF64::RowsAtCompileTime;

  //   // Optimization parameters.
  //   LieGroupF64 t_wr = t_w_init;
  //   PointF64 point_in_foo = point_a_init;

  //   // Build the problem.
  //   ::ceres::Problem problem;

  //   // Specify local update rule for our parameter

  //   auto parameterization = new sophus::ceres::Manifold<TLieGroup>;
  //   problem.AddParameterBlock(
  //       t_wr.unsafeMutPtr(), kNumParams, parameterization);

  //   // Create and add cost functions. Derivatives will be evaluated via
  //   // automatic differentiation
  //   ::ceres::CostFunction* cost_function1 = new
  //   ::ceres::AutoDiffCostFunction<
  //       TestLieGroupCostFunctor,
  //       LieGroupF64::kDof,
  //       LieGroupF64::kNumParams>(
  //       new TestLieGroupCostFunctor(t_w_targ.inverse()));
  //   problem.AddResidualBlock(cost_function1, nullptr, t_wr.unsafeMutPtr());
  //   ::ceres::CostFunction* cost_function2 = new
  //   ::ceres::AutoDiffCostFunction<
  //       TestPointCostFunctor,
  //       kNumPointParameters,
  //       kNumParams,
  //       kNumPointParameters>(new TestPointCostFunctor(point_b));
  //   problem.AddResidualBlock(cost_function2, nullptr, point_in_foo.data());

  //   // Set solver options (precision / method)
  //   ::ceres::Solver::Options options;
  //   options.gradient_tolerance = 0.01 * sophus::kEpsilonF64;
  //   options.function_tolerance = 0.01 * sophus::kEpsilonF64;
  //   options.linear_solver_type = ::ceres::DENSE_QR;
  //   options.max_num_iterations = 100;

  //   // Solve
  //   ::ceres::Solver::Summary summary;
  //   Solve(options, &problem, &summary);

  //   // Difference between target and parameter
  //   double const mse = (t_w_targ.inverse() * t_wr).log().squaredNorm();
  //   bool const passed = mse < 10. * sophus::kEpsilonF64;
  //   return passed;
  // }

  LieGroupCeresTests(
      std::vector<LieGroupF64> const& group_vec,
      std::vector<PointF64> const& point_vec)
      : group_vec(group_vec), point_vec(point_vec) {}

  std::vector<LieGroupF64> group_vec;
  std::vector<PointF64> point_vec;
};

TEST(sophus_ceres, regression) {
  LieGroupCeresTests<sophus::Rotation3>(
      sophus::Rotation3F64::elementExamples(),
      sophus::pointExamples<double, 3>())
      .testAll();

  LieGroupCeresTests<sophus::Isometry3>(
      sophus::Isometry3F64::elementExamples(),
      sophus::pointExamples<double, 3>())
      .testAll();

  LieGroupCeresTests<sophus::Rotation2>(
      sophus::Rotation2F64::elementExamples(),
      sophus::pointExamples<double, 2>())
      .testAll();

  LieGroupCeresTests<sophus::Isometry2>(
      sophus::Isometry2F64::elementExamples(),
      sophus::pointExamples<double, 2>())
      .testAll();

  LieGroupCeresTests<sophus::SpiralSimilarity3>(
      sophus::SpiralSimilarity3F64::elementExamples(),
      sophus::pointExamples<double, 3>())
      .testAll();

  LieGroupCeresTests<sophus::Similarity3>(
      sophus::Similarity3F64::elementExamples(),
      sophus::pointExamples<double, 3>())
      .testAll();

  LieGroupCeresTests<sophus::SpiralSimilarity2>(
      sophus::SpiralSimilarity2F64::elementExamples(),
      sophus::pointExamples<double, 2>())
      .testAll();

  LieGroupCeresTests<sophus::Similarity2>(
      sophus::Similarity2F64::elementExamples(),
      sophus::pointExamples<double, 2>())
      .testAll();
}

}  // namespace sophus::test
