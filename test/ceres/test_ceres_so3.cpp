/**
File adapted from Sophus

Copyright 2011-2017 Hauke Strasdat
          2012-2017 Steven Lovegrove

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
*/

#include <ceres/ceres.h>
#include <iostream>
#include <sophus/so3.hpp>

#include "local_parameterization_so3.hpp"


struct TestCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TestCostFunctor(Sophus::SO3d T_aw) : T_aw(T_aw) {}

  template <class T>
  bool operator()(T const* const sT_wa, T* sResiduals) const {
    Eigen::Map<Sophus::SO3<T> const> const T_wa(sT_wa);
    Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals(sResiduals);
    residuals = (T_aw.cast<T>() * T_wa).log();
    return true;
  }

  Sophus::SO3d T_aw;
};

bool test(Sophus::SO3d const& T_w_targ, Sophus::SO3d const& T_w_init) {
  Sophus::SO3d T_wr = T_w_init;
  ceres::Problem problem;
  // Specify local update rule for our parameter
  // EigenQuaternionParameterization fails at exact `pi`
  problem.AddParameterBlock(T_wr.data(), Sophus::SO3d::num_parameters,
                            new Sophus::test::LocalParameterizationSO3);

  TestCostFunctor* c = new TestCostFunctor(T_w_targ.inverse());
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<TestCostFunctor, Sophus::SO3d::DoF,
                                      Sophus::SO3d::num_parameters>(c);
  problem.AddResidualBlock(cost_function, NULL, T_wr.data());

  ceres::Solver::Options options;
  options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.linear_solver_type = ceres::DENSE_QR;

  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  // Difference between target and parameter
  double const mse = (T_w_targ.inverse() * T_wr).log().squaredNorm();
  bool const passed = mse < 10. * Sophus::Constants<double>::epsilon();
  return passed;
}

int main(int, char**) {
  using SO3Type = Sophus::SO3<double>;
  using Point = SO3Type::Point;
  double const kPi = Sophus::Constants<double>::pi();

  std::vector<SO3Type> so3_vec;
  so3_vec.push_back(SO3Type::exp(Point(0.1, 0.05, -0.7)));

  Point x = Point(0.33, -1., 1.5).normalized();
  so3_vec.push_back(SO3Type::exp(Point(0.1, 0.05, -0.7)) *
                    SO3Type::exp(x * (kPi - 1e-10)));
  so3_vec.push_back(SO3Type::exp(Point(0.1, 0.05, -0.7)) *
                    SO3Type::exp(x * (kPi - 1e-11)));
  so3_vec.push_back(SO3Type::exp(Point(0.1, 0.05, -0.7)) *
                    SO3Type::exp(x * (kPi)));

  for (size_t i = 1; i < so3_vec.size(); ++i) {
    bool const passed = test(so3_vec[i], so3_vec[0]);
    if (!passed) {
      std::cerr << "failed!" << std::endl << std::endl;
      exit(-1);
    }
  }

  return 0;
}
