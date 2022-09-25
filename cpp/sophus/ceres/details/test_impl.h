// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/ceres/manifold.h"
#include "sophus/common/test_macros.h"

template <class TLieGroup>
struct RotationalPart;

namespace sophus {

template <int kMatrixDim>
double dot(
    Eigen::Vector<double, kMatrixDim> const& v1,
    Eigen::Vector<double, kMatrixDim> const& v2) {
  return v1.dot(v2);
}

double dot(double const& a, double const& b) { return a * b; }

template <int kMatrixDim>
double squaredNorm(Eigen::Vector<double, kMatrixDim> const& vec) {
  return vec.squaredNorm();
}

double squaredNorm(double const& scalar) { return scalar * scalar; }

template <class TScalar>
TScalar zero() {
  return TScalar::Zero();
}

template <>
double zero<double>() {
  return 0.;
}

template <class TScalar>
typename TScalar::Scalar* data(TScalar& t) {
  return t.data();
}

double* data(double& d) { return &d; }

template <class TScalar>
const typename TScalar::Scalar* data(TScalar const& t) {
  return t.data();
}

double const* data(double const& d) { return &d; }

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

template <>
struct Random<double> {
  using TScalar = double;

  template <class TRt>
  static TScalar sample(TRt& rng) {
    std::normal_distribution<double> rnorm;
    return rnorm(rng);
  }
};

template <template <typename, int = 0> class TLieGroup>
struct LieGroupCeresTests {
  template <class TScalar>
  using LieGroup = TLieGroup<TScalar>;
  using LieGroupF64 = LieGroup<double>;
  using PointF64 = typename LieGroupF64::Point;
  using Tangentd = typename LieGroupF64::Tangent;
  template <class TScalar>
  using StdVector = std::vector<TScalar, Eigen::aligned_allocator<TScalar>>;
  static int constexpr kMatrixDim = LieGroupF64::kMatrixDim;
  static int constexpr kNumParameters = LieGroupF64::kNumParameters;
  static int constexpr kDoF = LieGroupF64::kDoF;

  struct TestLieGroupCostFunctor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TestLieGroupCostFunctor(LieGroupF64 const& t_aw) : t_aw(t_aw) {}

    template <class TScalar>
    bool operator()(TScalar const* const s_t_wa, TScalar* s_residuals) const {
      Eigen::Map<LieGroup<TScalar> const> const t_wa(s_t_wa);
      // Mapper class is only used to facciliate difference between
      // So2 (which uses Scalar as tangent vector type) and other groups
      // (which use Eigen::Vector<...> as tangent vector type).
      //
      // Feel free to use direct dereferencing or Eigen::Map depending
      // on ypur use-case for concrete application
      //
      // We only use Mapper class in order to make tests universally
      // compatible with LieGroup::Tangent being Scalar or Eigen::Vector
      using Mapper = Mapper<typename LieGroup<TScalar>::Tangent>;
      typename Mapper::Map residuals = Mapper::map(s_residuals);

      // We are able to mix Sophus types with doubles and Jet types withou
      // needing to cast to TScalar.
      residuals = (t_aw * t_wa).log();
      // Reverse order of multiplication. This forces the compiler to verify
      // that (Jet, double) and (double, Jet) LieGroup multiplication work
      // correctly.
      residuals = (t_wa * t_aw).log();
      // Finally, ensure that Jet-to-Jet multiplication works.
      residuals = (t_wa * t_aw.template cast<TScalar>()).log();
      return true;
    }

    LieGroupF64 t_aw;
  };
  struct TestPointCostFunctor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TestPointCostFunctor(LieGroupF64 const& t_aw, PointF64 const& point_a)
        : t_aw(t_aw), point_a(point_a) {}

    template <class TScalar>
    bool operator()(
        TScalar const* const s_t_wa,
        TScalar const* const spoint_b,
        TScalar* s_residuals) const {
      using LieGroup = TLieGroup<TScalar>;
      using Point = typename LieGroup::Point;
      Eigen::Map<LieGroup const> const t_wa(s_t_wa);
      Eigen::Map<Point const> point_b(spoint_b);
      Eigen::Map<Point> residuals(s_residuals);

      // Multiply LieGroupF64 by Jet vector3.
      Point point_b_prime = t_aw * point_b;
      // Ensure Jet LieGroup multiplication with Jet vector3.
      point_b_prime = t_aw.template cast<TScalar>() * point_b;

      // Multiply Jet LieGroup with Vector3d.
      Point point_a_prime = t_wa * point_a;
      // Ensure Jet LieGroup multiplication with Jet vector3.
      point_a_prime = t_wa * point_a.template cast<TScalar>();

      residuals = point_b_prime - point_a_prime;
      return true;
    }

    LieGroupF64 t_aw;
    PointF64 point_a;
  };

  struct TestGraphFunctor {
    template <class TScalar>
    bool operator()(
        TScalar const* raw_a, TScalar const* raw_b, TScalar* residuals) const {
      using LieGroup = TLieGroup<TScalar>;
      Eigen::Map<const LieGroup> a(raw_a);
      Eigen::Map<const LieGroup> b(raw_b);
      using Mapper = Mapper<typename LieGroup::Tangent>;
      typename Mapper::Map diff_log = Mapper::map(residuals);

      // Jet LieGroup multiplication with LieGroupF64
      diff_log = (diff * (b.inverse() * a)).log();
      return true;
    }

    TestGraphFunctor(LieGroupF64 const& diff) : diff(diff) {}
    const LieGroupF64 diff;
  };

  bool testAll() {
    bool passed = true;
    for (size_t i = 0; i < group_vec.size(); ++i) {
      for (size_t j = 0; j < group_vec.size(); ++j) {
        if (i == j) {
          continue;
        }
        for (size_t k = 0; k < point_vec.size(); ++k) {
          for (size_t l = 0; l < point_vec.size(); ++l) {
            if (k == l) {
              continue;
            }
            std::cerr << "Simple test #" << i << ", " << j << ", " << k << ", "
                      << l;
            passed &=
                test(group_vec[i], group_vec[j], point_vec[k], point_vec[l]);
            processTestResult(passed);
          }
        }
      }
    }
    for (size_t i = 0; i < group_vec.size(); ++i) {
      for (size_t j = 0; j < group_vec.size(); ++j) {
        passed &= testManifold(group_vec[i], group_vec[j]);
        processTestResult(passed);
      }
    }
    int ns[] = {20, 40, 80, 160};
    for (auto k_matrix_dim : ns) {
      std::cerr << "Averaging test: kMatrixDim = " << k_matrix_dim;
      passed &= testAveraging(k_matrix_dim, .5, .1);
      processTestResult(passed);
    }
    return passed;
  }

  bool testAveraging(
      const size_t num_vertices,
      double const sigma_init,
      double const sigma_observation) {
    if (num_vertices == 0u) {
      return true;
    }
    double const sigma_init_elementwise = sigma_init / std::sqrt(kDoF);
    double const sigma_observation_elementwise =
        sigma_observation / std::sqrt(kDoF);
    // Running Lie group averaging on a K_n graph with a random initialization
    // noise and random noise in observations
    ceres::Problem problem;

    // "Random" initialization in order to keep tests repeatable
    std::mt19937 rng(2021);
    StdVector<LieGroupF64> vec(num_vertices);

    StdVector<LieGroupF64> v_estimate;
    v_estimate.reserve(num_vertices);
    double initial_error = 0.;
    auto parametrization = new sophus::Manifold<TLieGroup>;

    // All vertices are initialized with an i.i.d noise with normal
    // distribution; Scaling is adjusted in order to maintain the same
    // expectation of squared norm for all groups
    for (size_t i = 0; i < num_vertices; ++i) {
      auto& v = vec[i];
      v = LieGroupF64::sampleUniform(rng);
      const Tangentd delta_log =
          Random<Tangentd>::sample(rng) * sigma_init_elementwise;
      const LieGroupF64 delta = LieGroupF64::exp(delta_log);
      v_estimate.emplace_back(v * delta);
      initial_error += squaredNorm(delta_log);
      problem.AddParameterBlock(
          v_estimate.back().data(),
          LieGroupF64::kNumParameters,
          parametrization);
    }

    // For simplicity of graph generation, we use a complete (undirected) graph.
    // Each edge (observation) has i.i.d noise with multivariate normal
    // distribution; Scaling is adjusted in order to maintain the same
    // expectation of squared norm for all groups
    for (size_t i = 0; i < num_vertices; ++i) {
      for (size_t j = i + 1; j < num_vertices; ++j) {
        LieGroupF64 diff = vec[i].inverse() * vec[j];
        auto const delta_log =
            Random<typename LieGroupF64::Tangent>::sample(rng) *
            sigma_observation_elementwise;
        auto const delta = LieGroupF64::exp(delta_log);
        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<
            TestGraphFunctor,
            LieGroupF64::kDoF,
            LieGroupF64::kNumParameters,
            LieGroupF64::kNumParameters>(new TestGraphFunctor(diff * delta));
        // For real-world problems you should consider using robust
        // loss-function
        problem.AddResidualBlock(
            cost, nullptr, v_estimate[i].data(), v_estimate[j].data());
      }
    }

    ceres::Solver::Options options;
    options.gradient_tolerance = 1e-2 * sophus::kEpsilonF64;
    options.function_tolerance = 1e-2 * sophus::kEpsilonF64;
    options.parameter_tolerance = 1e-2 * sophus::kEpsilonF64;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    // Computing final error in the estimates
    double final_error = 0.;
    for (size_t i = 0; i < num_vertices; ++i) {
      final_error += squaredNorm((vec[i].inverse() * v_estimate[i]).log());
    }

    // Expecting reasonable decrease of both estimates' errors and residuals
    return summary.final_cost < .25 * summary.initial_cost &&
           final_error < .25 * initial_error;
  }

  bool test(
      LieGroupF64 const& t_w_targ,
      LieGroupF64 const& t_w_init,
      PointF64 const& point_a_init,
      PointF64 const& point_b) {
    static int constexpr kNumPointParameters = PointF64::RowsAtCompileTime;

    // Optimization parameters.
    LieGroupF64 t_wr = t_w_init;
    PointF64 point_a = point_a_init;

    // Build the problem.
    ceres::Problem problem;

    // Specify local update rule for our parameter

    auto parameterization = new sophus::Manifold<TLieGroup>;
    problem.AddParameterBlock(t_wr.data(), kNumParameters, parameterization);

    // Create and add cost functions. Derivatives will be evaluated via
    // automatic differentiation
    ceres::CostFunction* cost_function1 = new ceres::AutoDiffCostFunction<
        TestLieGroupCostFunctor,
        LieGroupF64::kDoF,
        LieGroupF64::kNumParameters>(
        new TestLieGroupCostFunctor(t_w_targ.inverse()));
    problem.AddResidualBlock(cost_function1, nullptr, t_wr.data());
    ceres::CostFunction* cost_function2 = new ceres::AutoDiffCostFunction<
        TestPointCostFunctor,
        kNumPointParameters,
        kNumParameters,
        kNumPointParameters>(
        new TestPointCostFunctor(t_w_targ.inverse(), point_b));
    problem.AddResidualBlock(
        cost_function2, nullptr, t_wr.data(), point_a.data());

    // Set solver options (precision / method)
    ceres::Solver::Options options;
    options.gradient_tolerance = 0.01 * sophus::kEpsilonF64;
    options.function_tolerance = 0.01 * sophus::kEpsilonF64;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;

    // Solve
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    // Difference between target and parameter
    double const mse = squaredNorm((t_w_targ.inverse() * t_wr).log());
    bool const passed = mse < 10. * sophus::kEpsilonF64;
    return passed;
  }

  bool testManifold(LieGroupF64 const& x, LieGroupF64 const& y) {
    // ceres/manifold_test_utils.h is google-test based; here we check all the
    // same invariants
    const Tangentd delta = (x.inverse() * y).log();
    const Tangentd o = zero<Tangentd>();
    sophus::Manifold<TLieGroup> manifold;

    LieGroupF64 test_group;

    bool passed = true;
    auto coeffs =
        Eigen::Map<const Eigen::Matrix<double, kNumParameters, 1>>(x.data());
    auto coeffs_y =
        Eigen::Map<const Eigen::Matrix<double, kNumParameters, 1>>(y.data());
    std::cerr << "XPlusZeroIsXAt " << coeffs.transpose() << std::endl;
    passed &= xPlusZeroIsXAt(x);
    std::cerr << "XMinusXIsZeroAt " << coeffs.transpose() << std::endl;
    passed &= xMinusXIsZeroAt(x);
    std::cerr << "MinusPlusIsIdentityAt " << coeffs.transpose() << std::endl;
    passed &= minusPlusIsIdentityAt(x, delta);
    std::cerr << "MinusPlusIsIdentityAt " << coeffs.transpose() << std::endl;
    passed &= minusPlusIsIdentityAt(x, o);
    std::cerr << "PlusMinusIsIdentityAt " << coeffs.transpose() << std::endl;
    passed &= plusMinusIsIdentityAt(x, x);
    std::cerr << "PlusMinusIsIdentityAt " << coeffs.transpose() << " "
              << coeffs_y.transpose() << std::endl;
    passed &= plusMinusIsIdentityAt(x, y);
    std::cerr << "MinusPlusJacobianIsIdentityAt " << coeffs.transpose()
              << std::endl;
    passed &= minusPlusJacobianIsIdentityAt(x);
    return passed;
  }

  bool xPlusZeroIsXAt(LieGroupF64 const& x) {
    const Tangentd o = zero<Tangentd>();
    sophus::Manifold<TLieGroup> manifold;
    LieGroupF64 test_group;

    bool passed = true;

    passed &= manifold.Plus(x.data(), data(o), test_group.data());
    processTestResult(passed);
    double const error = squaredNorm((x.inverse() * test_group).log());
    passed &= error < sophus::kEpsilonF64;
    processTestResult(passed);
    return passed;
  }

  bool xMinusXIsZeroAt(LieGroupF64 const& x) {
    sophus::Manifold<TLieGroup> manifold;
    LieGroupF64 test_group;
    Tangentd test_tangent;

    bool passed = true;

    passed &= manifold.Minus(x.data(), x.data(), data(test_tangent));
    processTestResult(passed);
    double const error = squaredNorm(test_tangent);
    passed &= error < sophus::kEpsilonF64;
    processTestResult(passed);
    return passed;
  }

  bool minusPlusIsIdentityAt(LieGroupF64 const& x, Tangentd const& delta) {
    if (RotationalPart<LieGroupF64>::norm(delta) >
        sophus::kPi<double> * (1. - sophus::kEpsilonF64)) {
      return true;
    }
    sophus::Manifold<TLieGroup> manifold;
    LieGroupF64 test_group;
    Tangentd test_tangent;

    bool passed = true;

    passed &= manifold.Plus(x.data(), data(delta), test_group.data());
    processTestResult(passed);

    passed &= manifold.Minus(test_group.data(), x.data(), data(test_tangent));
    processTestResult(passed);

    const Tangentd diff = test_tangent - delta;
    double const error = squaredNorm(diff);
    passed &= error < sophus::kEpsilonF64;
    processTestResult(passed);
    return passed;
  }

  bool plusMinusIsIdentityAt(LieGroupF64 const& x, LieGroupF64 const& y) {
    sophus::Manifold<TLieGroup> manifold;
    LieGroupF64 test_group;
    Tangentd test_tangent;

    bool passed = true;

    passed &= manifold.Minus(y.data(), x.data(), data(test_tangent));
    processTestResult(passed);

    passed &= manifold.Plus(x.data(), data(test_tangent), test_group.data());
    processTestResult(passed);

    double const error = squaredNorm((y.inverse() * test_group).log());
    passed &= error < sophus::kEpsilonF64;
    processTestResult(passed);
    return passed;
  }

  bool minusPlusJacobianIsIdentityAt(LieGroupF64 const& x) {
    sophus::Manifold<TLieGroup> manifold;
    LieGroupF64 test_group;

    bool passed = true;

    Eigen::Matrix<
        double,
        kNumParameters,
        kDoF,
        kDoF == 1 ? Eigen::ColMajor : Eigen::RowMajor>
        jplus;
    Eigen::Matrix<double, kDoF, kNumParameters, Eigen::RowMajor> jminus;

    passed &= manifold.PlusJacobian(x.data(), jplus.data());
    processTestResult(passed);

    passed &= manifold.MinusJacobian(x.data(), jminus.data());
    processTestResult(passed);

    const Eigen::Matrix<double, kDoF, kDoF> diff =
        jminus * jplus - Eigen::Matrix<double, kDoF, kDoF>::Identity();

    std::cerr << diff << std::endl;
    double const error = diff.squaredNorm();
    passed &= error < sophus::kEpsilonF64;
    processTestResult(passed);
    return passed;
  }

  LieGroupCeresTests(
      StdVector<LieGroupF64> const& group_vec,
      StdVector<PointF64> const& point_vec)
      : group_vec(group_vec), point_vec(point_vec) {}

  StdVector<LieGroupF64> group_vec;
  StdVector<PointF64> point_vec;
};

}  // namespace sophus
