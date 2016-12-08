#include <ceres/ceres.h>
#include <iostream>
#include <sophus/se3.hpp>

#include "local_parameterization_se3.hpp"

struct TestCostFunctor {
  TestCostFunctor(Sophus::SE3d T_aw) : T_aw(T_aw) {}

  template <typename T>
  bool operator()(const T* const sT_wa, T* sResiduals) const {
    const Eigen::Map<const Sophus::SE3Group<T> > T_wa(sT_wa);
    Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(sResiduals);

    residuals = (T_aw.cast<T>() * T_wa).log();
    return true;
  }

  Sophus::SE3d T_aw;
};

bool test(const Sophus::SE3d& T_w_targ, const Sophus::SE3d& T_w_init) {
  // Optimisation parameter
  Sophus::SE3d T_wr = T_w_init;

  // Build the problem.
  ceres::Problem problem;

  // Specify local update rule for our parameter
  problem.AddParameterBlock(T_wr.data(), Sophus::SE3d::num_parameters,
                            new Sophus::test::LocalParameterizationSE3);

  // Create and add cost function. Derivatives will be evaluated via
  // automatic differentiation

  TestCostFunctor* c = new TestCostFunctor(T_w_targ.inverse());
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<TestCostFunctor, Sophus::SE3d::DoF,
                                      Sophus::SE3d::num_parameters>(c);
  problem.AddResidualBlock(cost_function, NULL, T_wr.data());

  // Set solver options (precision / method)
  ceres::Solver::Options options;
  options.gradient_tolerance =
      0.01 * Sophus::Constants<double>::epsilon();
  options.function_tolerance =
      0.01 * Sophus::Constants<double>::epsilon();
  options.linear_solver_type = ceres::DENSE_QR;

  // Solve
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  // Difference between target and parameter
  const double mse = (T_w_targ.inverse() * T_wr).log().squaredNorm();
  const bool passed = mse < 10. * Sophus::Constants<double>::epsilon();
  return passed;
}

int main(int, char**) {
  typedef Sophus::SE3Group<double> SE3Type;
  typedef Sophus::SO3Group<double> SO3Type;
  typedef SE3Type::Point Point;
  const double PI = Constants<double>::pi();

  std::vector<SE3Type> se3_vec;
  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)));
  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0.2, 0.5, -1.0)), Point(10, 0, 0)));
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.)), Point(0, 100, 5)));
  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0., 0., 0.00001)), Point(0, 0, 0)));
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(0., 0., 0.00001)),
                            Point(0, -0.00000001, 0.0000000001)));
  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0., 0., 0.00001)), Point(0.01, 0, 0)));
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(PI, 0, 0)), Point(4, -5, 0)));
  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)) *
      SE3Type(SO3Type::exp(Point(PI, 0, 0)), Point(0, 0, 0)) *
      SE3Type(SO3Type::exp(Point(-0.2, -0.5, -0.0)), Point(0, 0, 0)));
  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0.3, 0.5, 0.1)), Point(2, 0, -7)) *
      SE3Type(SO3Type::exp(Point(PI, 0, 0)), Point(0, 0, 0)) *
      SE3Type(SO3Type::exp(Point(-0.3, -0.5, -0.1)), Point(0, 6, 0)));

  for (size_t i = 0; i < se3_vec.size(); ++i) {
    const bool passed = test(se3_vec[i], se3_vec[(i + 3) % se3_vec.size()]);
    if (!passed) {
      std::cerr << "failed!" << std::endl << std::endl;
      exit(-1);
    }
  }

  return 0;
}
