#ifndef SOPHUS_CERES_TESTS_HPP
#define SOPHUS_CERES_TESTS_HPP

#include <ceres/ceres.h>

#include <sophus/ceres_local_parameterization.hpp>

namespace Sophus {

template <int N>
double squaredNorm(const Vector<double, N>& vec) {
  return vec.squaredNorm();
}

double squaredNorm(const double& scalar) { return scalar * scalar; }

template <template <typename, int = 0> class LieGroup_>
struct LieGroupCeresTests {
  template <typename T>
  using LieGroup = LieGroup_<T>;
  using LieGroupd = LieGroup<double>;
  using Pointd = typename LieGroupd::Point;
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
      Eigen::Map<typename LieGroup<T>::Tangent> residuals(sResiduals);

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

  bool testAll() {
    bool passed = true;
    for (size_t i = 0; i < group_vec.size(); ++i) {
      for (size_t j = 0; j < group_vec.size(); ++j) {
        if (i == j) continue;
        for (size_t k = 0; k < point_vec.size(); ++k) {
          for (size_t l = 0; l < point_vec.size(); ++l) {
            if (k == l) continue;
            passed &=
                test(group_vec[i], group_vec[j], point_vec[k], point_vec[l]);
          }
        }
      }
    }
    return passed;
  }

  bool test(LieGroupd const& T_w_targ, LieGroupd const& T_w_init,
            Pointd const& point_a_init, Pointd const& point_b) {
    static constexpr int kNumPointParameters = Pointd::RowsAtCompileTime;

    // Optimisation parameters.
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
    problem.AddResidualBlock(cost_function1, NULL, T_wr.data());
    ceres::CostFunction* cost_function2 =
        new ceres::AutoDiffCostFunction<TestPointCostFunctor,
                                        kNumPointParameters, num_parameters,
                                        kNumPointParameters>(
            new TestPointCostFunctor(T_w_targ.inverse(), point_b));
    problem.AddResidualBlock(cost_function2, NULL, T_wr.data(), point_a.data());

    // Set solver options (precision / method)
    ceres::Solver::Options options;
    options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
    options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
    options.linear_solver_type = ceres::DENSE_QR;

    // Solve
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

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
