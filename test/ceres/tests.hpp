#ifndef SOPHUS_CERES_TESTS_HPP
#define SOPHUS_CERES_TESTS_HPP

#include <ceres/ceres.h>

#include <sophus/ceres_local_parameterization.hpp>
#include <sophus/test_macros.hpp>

namespace Sophus {

template <int N>
double dot(const Vector<double, N>& v1, const Vector<double, N>& v2) {
  return v1.dot(v2);
}

double dot(const double& a, const double& b) { return a * b; }

template <int N>
double squaredNorm(const Vector<double, N>& vec) {
  return vec.squaredNorm();
}

double squaredNorm(const double& scalar) { return scalar * scalar; }

template <typename T>
struct Random {
  template <typename R>
  T static sample(R& rng) {
    std::normal_distribution<double> rnorm;
    static_assert(T::RowsAtCompileTime >= 0,
                  "Matrix should have known size at compile-time");
    static_assert(T::ColsAtCompileTime >= 0,
                  "Matrix should have known size at compile-time");
    T res;
    for (Eigen::Index i = 0; i < res.size(); ++i) res.data()[i] = rnorm(rng);
    return res;
  }
};

template <>
struct Random<double> {
  using T = double;

  template <typename R>
  T static sample(R& rng) {
    std::normal_distribution<double> rnorm;
    return rnorm(rng);
  }
};

template <template <typename, int = 0> class LieGroup_>
struct LieGroupCeresTests {
  template <typename T>
  using LieGroup = LieGroup_<T>;
  using LieGroupd = LieGroup<double>;
  using Pointd = typename LieGroupd::Point;
  using Tangentd = typename LieGroupd::Tangent;
  template <typename T>
  using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;
  static int constexpr N = LieGroupd::N;
  static int constexpr num_parameters = LieGroupd::num_parameters;
  static int constexpr DoF = LieGroupd::DoF;

  struct TestLieGroupCostFunctor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TestLieGroupCostFunctor(const LieGroupd& T_aw) : T_aw(T_aw) {}

    template <class T>
    bool operator()(T const* const sT_wa, T* sResiduals) const {
      Eigen::Map<LieGroup<T> const> const T_wa(sT_wa);
      // Mapper class is only used to facciliate difference between
      // SO2 (which uses Scalar as tangent vector type) and other groups
      // (which use Vector<...> as tangent vector type).
      //
      // Feel free to use direct dereferencing or Eigen::Map depending
      // on ypur use-case for concrete application
      //
      // We only use Mapper class in order to make tests universally
      // compatible with LieGroup::Tangent being Scalar or Vector
      using Mapper = Mapper<typename LieGroup<T>::Tangent>;
      typename Mapper::Map residuals = Mapper::map(sResiduals);

      // We are able to mix Sophus types with doubles and Jet types withou
      // needing to cast to T.
      residuals = (T_aw * T_wa).log();
      // Reverse order of multiplication. This forces the compiler to verify
      // that (Jet, double) and (double, Jet) LieGroup multiplication work
      // correctly.
      residuals = (T_wa * T_aw).log();
      // Finally, ensure that Jet-to-Jet multiplication works.
      residuals = (T_wa * T_aw.template cast<T>()).log();
      return true;
    }

    LieGroupd T_aw;
  };
  struct TestPointCostFunctor {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TestPointCostFunctor(const LieGroupd& T_aw, const Pointd& point_a)
        : T_aw(T_aw), point_a(point_a) {}

    template <class T>
    bool operator()(T const* const sT_wa, T const* const spoint_b,
                    T* sResiduals) const {
      using LieGroupT = LieGroup<T>;
      using PointT = typename LieGroupT::Point;
      Eigen::Map<LieGroupT const> const T_wa(sT_wa);
      Eigen::Map<PointT const> point_b(spoint_b);
      Eigen::Map<PointT> residuals(sResiduals);

      // Multiply LieGroupd by Jet Vector3.
      PointT point_b_prime = T_aw * point_b;
      // Ensure Jet LieGroup multiplication with Jet Vector3.
      point_b_prime = T_aw.template cast<T>() * point_b;

      // Multiply Jet LieGroup with Vector3d.
      PointT point_a_prime = T_wa * point_a;
      // Ensure Jet LieGroup multiplication with Jet Vector3.
      point_a_prime = T_wa * point_a.template cast<T>();

      residuals = point_b_prime - point_a_prime;
      return true;
    }

    LieGroupd T_aw;
    Pointd point_a;
  };

  struct TestGraphFunctor {
    template <typename T>
    bool operator()(const T* a, const T* b, T* residuals) const {
      using LieGroupT = LieGroup<T>;
      Eigen::Map<const LieGroupT> A(a);
      Eigen::Map<const LieGroupT> B(b);
      using Mapper = Mapper<typename LieGroupT::Tangent>;
      typename Mapper::Map diff_log = Mapper::map(residuals);

      // Jet LieGroup multiplication with LieGroupd
      diff_log = (diff * (B.inverse() * A)).log();
      return true;
    }

    TestGraphFunctor(const LieGroupd& diff) : diff(diff) {}
    const LieGroupd diff;
  };

  bool testAll() {
    bool passed = true;
    for (size_t i = 0; i < group_vec.size(); ++i) {
      for (size_t j = 0; j < group_vec.size(); ++j) {
        if (i == j) continue;
        for (size_t k = 0; k < point_vec.size(); ++k) {
          for (size_t l = 0; l < point_vec.size(); ++l) {
            if (k == l) continue;
            std::cerr << "Simple test #" << i << ", " << j << ", " << k << ", "
                      << l;
            passed &=
                test(group_vec[i], group_vec[j], point_vec[k], point_vec[l]);
            processTestResult(passed);
          }
        }
      }
    }
    int Ns[] = {20, 40, 80, 160};
    for (auto N : Ns) {
      std::cerr << "Averaging test: N = " << N;
      passed &= testAveraging(N, .5, .1);
      processTestResult(passed);
    }
    return passed;
  }

  bool testAveraging(const size_t num_vertices, const double sigma_init,
                     const double sigma_observation) {
    if (!num_vertices) return true;
    const double sigma_init_elementwise = sigma_init / std::sqrt(DoF);
    const double sigma_observation_elementwise =
        sigma_observation / std::sqrt(DoF);
    // Running Lie group averaging on a K_n graph with a random initialization
    // noise and random noise in observations
    ceres::Problem problem;

    // "Random" initialization in order to keep tests repeatable
    std::mt19937 rng(2021);
    StdVector<LieGroupd> V(num_vertices), V_estimate;
    V_estimate.reserve(num_vertices);
    double initial_error = 0.;
    // All parameter blocks have the same parameterization, thus it can be
    // reused. Memory will be freed by ceres-solver
    auto parametrization = new Sophus::LocalParameterization<LieGroup_>;
    // All vertices are initialized with an i.i.d noise with normal
    // distribution; Scaling is adjusted in order to maintain the same
    // expectation of squared norm for all groups
    for (size_t i = 0; i < num_vertices; ++i) {
      auto& v = V[i];
      v = LieGroupd::sampleUniform(rng);
      const Tangentd delta_log =
          Random<Tangentd>::sample(rng) * sigma_init_elementwise;
      const LieGroupd delta = LieGroupd::exp(delta_log);
      V_estimate.emplace_back(v * delta);
      initial_error += squaredNorm(delta_log);
      problem.AddParameterBlock(V_estimate.back().data(),
                                LieGroupd::num_parameters, parametrization);
    }

    // For simplicity of graph generation, we use a complete (undirected) graph.
    // Each edge (observation) has i.i.d noise with multivariate normal
    // distribution; Scaling is adjusted in order to maintain the same
    // expectation of squared norm for all groups
    for (size_t i = 0; i < num_vertices; ++i)
      for (size_t j = i + 1; j < num_vertices; ++j) {
        LieGroupd diff = V[i].inverse() * V[j];
        const auto delta_log =
            Random<typename LieGroupd::Tangent>::sample(rng) *
            sigma_observation_elementwise;
        const auto delta = LieGroupd::exp(delta_log);
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<TestGraphFunctor, LieGroupd::DoF,
                                            LieGroupd::num_parameters,
                                            LieGroupd::num_parameters>(
                new TestGraphFunctor(diff * delta));
        // For real-world problems you should consider using robust
        // loss-function
        problem.AddResidualBlock(cost, nullptr, V_estimate[i].data(),
                                 V_estimate[j].data());
      }

    ceres::Solver::Options options;
    options.gradient_tolerance = 1e-2 * Sophus::Constants<double>::epsilon();
    options.function_tolerance = 1e-2 * Sophus::Constants<double>::epsilon();
    options.parameter_tolerance = 1e-2 * Sophus::Constants<double>::epsilon();
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    // Computing final error in the estimates
    double final_error = 0.;
    for (size_t i = 0; i < num_vertices; ++i) {
      final_error += squaredNorm((V[i].inverse() * V_estimate[i]).log());
    }

    // Expecting reasonable decrease of both estimates' errors and residuals
    return summary.final_cost < .25 * summary.initial_cost &&
           final_error < .25 * initial_error;
  }

  bool test(LieGroupd const& T_w_targ, LieGroupd const& T_w_init,
            Pointd const& point_a_init, Pointd const& point_b) {
    static constexpr int kNumPointParameters = Pointd::RowsAtCompileTime;

    // Optimization parameters.
    LieGroupd T_wr = T_w_init;
    Pointd point_a = point_a_init;

    // Build the problem.
    ceres::Problem problem;

    // Specify local update rule for our parameter
    problem.AddParameterBlock(T_wr.data(), num_parameters,
                              new Sophus::LocalParameterization<LieGroup_>);

    // Create and add cost functions. Derivatives will be evaluated via
    // automatic differentiation
    ceres::CostFunction* cost_function1 =
        new ceres::AutoDiffCostFunction<TestLieGroupCostFunctor, LieGroupd::DoF,
                                        LieGroupd::num_parameters>(
            new TestLieGroupCostFunctor(T_w_targ.inverse()));
    problem.AddResidualBlock(cost_function1, nullptr, T_wr.data());
    ceres::CostFunction* cost_function2 =
        new ceres::AutoDiffCostFunction<TestPointCostFunctor,
                                        kNumPointParameters, num_parameters,
                                        kNumPointParameters>(
            new TestPointCostFunctor(T_w_targ.inverse(), point_b));
    problem.AddResidualBlock(cost_function2, nullptr, T_wr.data(),
                             point_a.data());

    // Set solver options (precision / method)
    ceres::Solver::Options options;
    options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
    options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;

    // Solve
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    // Difference between target and parameter
    double const mse = squaredNorm((T_w_targ.inverse() * T_wr).log());
    bool const passed = mse < 10. * Sophus::Constants<double>::epsilon();
    return passed;
  }

  LieGroupCeresTests(const StdVector<LieGroupd>& group_vec,
                     const StdVector<Pointd>& point_vec)
      : group_vec(group_vec), point_vec(point_vec) {}

  StdVector<LieGroupd> group_vec;
  StdVector<Pointd> point_vec;
};

}  // namespace Sophus

#endif
