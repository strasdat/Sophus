#include <ceres/ceres.h>
#include <iostream>
#include <sophus/so3.hpp>

struct TestCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TestCostFunctor(Sophus::SO3d C_aw) : C_aw(C_aw) {}

  template <class T>
  bool operator()(T const* const sC_wa, T* sResiduals) const {
    Eigen::Map<Sophus::SO3<T> const> const C_wa(sC_wa);
    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals(sResiduals);

    // We are able to mix Sophus types with doubles and Jet types without
    // needing to cast to T.
    residuals = (C_aw * C_wa).log();
    // Reverse order of multiplication. This forces the compiler to verify that
    // (Jet, double) and (double, Jet) SO3 multiplication work correctly.
    residuals = (C_wa * C_aw).log();
    // Finally, ensure that Jet-to-Jet multiplication works.
    residuals = (C_wa * C_aw.cast<T>()).log();
    return true;
  }

  Sophus::SO3d C_aw;
};

// Checks if ceres optimization will proceed correctly given problematic or
// close-to-singular initial conditions, i.e. approx. 180-deg rotation, which
// trips a flaw in old implementation of SO3::log() where the tangent vector's
// magnitude is set to a constant close to \pi whenever the input rotation's
// rotation angle is within some tolerance of \pi, giving zero gradients wrt
// scalar part of quaternion.
bool test(Sophus::SO3d const& C_w_targ, Sophus::SO3d const& C_w_init) {
  Sophus::SO3d C_wr = C_w_init;
  ceres::Problem problem;
  // Specify local update rule for our parameter.
  problem.AddParameterBlock(C_wr.data(), Sophus::SO3d::num_parameters,
                            new ceres::EigenQuaternionParameterization);

  // Create and add cost functions. Derivatives will be evaluated via
  // automatic differentiation
  ceres::CostFunction* cosC_function =
      new ceres::AutoDiffCostFunction<TestCostFunctor, Sophus::SO3d::DoF,
                                      Sophus::SO3d::num_parameters>(
          new TestCostFunctor(C_w_targ.inverse()));
  problem.AddResidualBlock(cosC_function, NULL, C_wr.data());

  // Set solver options (precision / method)
  ceres::Solver::Options options;
  options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.linear_solver_type = ceres::DENSE_QR;

  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  // Difference between target and parameter
  double const mse = (C_w_targ.inverse() * C_wr).log().squaredNorm();
  bool const passed = mse < 10. * Sophus::Constants<double>::epsilon();
  return passed;
}

int main(int, char**) {
  using SO3Type = Sophus::SO3<double>;
  using Tangent = SO3Type::Tangent;
  double const kPi = Sophus::Constants<double>::pi();
  double const epsilon = Sophus::Constants<double>::epsilon();

  SO3Type C_0 = SO3Type::exp(Tangent(0.1, 0.05, -0.7));

  Tangent axis_0(0.18005924, -0.54563405, 0.81845107);
  auto ics = {C_0 * SO3Type::exp(axis_0 * 1.0),  // Generic rotation angle < pi
              C_0 * SO3Type::exp(axis_0 * -1.0),
              C_0 * SO3Type::exp(axis_0 * 4.0),  // Generic rotation angle > pi
              C_0 * SO3Type::exp(axis_0 * -4.0),
              C_0 * SO3Type::exp(axis_0 * kPi),  // Singular rotation angle = pi
              C_0 * SO3Type::exp(axis_0 * -kPi),
              C_0 * SO3Type::exp(axis_0 * kPi * (1.0 + epsilon)),
              C_0 * SO3Type::exp(axis_0 * kPi * (1.0 - epsilon)),
              C_0 * SO3Type::exp(axis_0 * -kPi * (1.0 + epsilon)),
              C_0 * SO3Type::exp(axis_0 * -kPi * (1.0 - epsilon))};
  // Now solve problems.
  for (const auto& it : ics) {
    bool const passed = test(C_0, it);
    if (!passed) {
      std::cerr << "failed!" << std::endl << std::endl;
      exit(-1);
    }
  }

  return 0;
}
