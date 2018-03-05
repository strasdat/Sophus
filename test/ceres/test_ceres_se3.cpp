#include <ceres/ceres.h>
#include <iostream>
#include <sophus/se3.hpp>

#include "local_parameterization_se3.hpp"

// Eigen's ostream operator is not compatible with ceres::Jet types.
// In particular, Eigen assumes that the scalar type (here Jet<T,N>) can be
// casted to an arithmetic type, which is not true for ceres::Jet.
// Unfortunatly, the ceres::Jet class does not define a conversion
// operator (http://en.cppreference.com/w/cpp/language/cast_operator).
//
// This workaround creates a template specilization for Eigen's cast_impl,
// when casting from a ceres::Jet type. It relies on Eigen's internal API and
// might break with future versions of Eigen.
namespace Eigen {
namespace internal {

template <class T, int N, typename NewType>
struct cast_impl<ceres::Jet<T, N>, NewType> {
  EIGEN_DEVICE_FUNC
  static inline NewType run(ceres::Jet<T, N> const& x) {
    return static_cast<NewType>(x.a);
  }
};

}  // namespace internal
}  // namespace Eigen

struct TestCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TestCostFunctor(Sophus::SE3d T_aw) : T_aw(T_aw) {}

  template <class T>
  bool operator()(T const* const sT_wa, T* sResiduals) const {
    Eigen::Map<Sophus::SE3<T> const> const T_wa(sT_wa);
    Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(sResiduals);

    residuals = (T_aw.cast<T>() * T_wa).log();
    return true;
  }

  Sophus::SE3d T_aw;
};

bool test(Sophus::SE3d const& T_w_targ, Sophus::SE3d const& T_w_init) {
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
  options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
  options.linear_solver_type = ceres::DENSE_QR;

  // Solve
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  // Difference between target and parameter
  double const mse = (T_w_targ.inverse() * T_wr).log().squaredNorm();
  bool const passed = mse < 10. * Sophus::Constants<double>::epsilon();
  return passed;
}

template <typename Scalar>
bool CreateSE3FromMatrix(Eigen::Matrix<Scalar, 4, 4> mat) {
  auto se3 = Sophus::SE3<Scalar>(mat);
  se3 = se3;
  return true;
}

int main(int, char**) {
  using SE3Type = Sophus::SE3<double>;
  using SO3Type = Sophus::SO3<double>;
  using Point = SE3Type::Point;
  double const kPi = Sophus::Constants<double>::pi();

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
  se3_vec.push_back(SE3Type(SO3Type::exp(Point(kPi, 0, 0)), Point(4, -5, 0)));
  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)) *
      SE3Type(SO3Type::exp(Point(kPi, 0, 0)), Point(0, 0, 0)) *
      SE3Type(SO3Type::exp(Point(-0.2, -0.5, -0.0)), Point(0, 0, 0)));
  se3_vec.push_back(
      SE3Type(SO3Type::exp(Point(0.3, 0.5, 0.1)), Point(2, 0, -7)) *
      SE3Type(SO3Type::exp(Point(kPi, 0, 0)), Point(0, 0, 0)) *
      SE3Type(SO3Type::exp(Point(-0.3, -0.5, -0.1)), Point(0, 6, 0)));

  for (size_t i = 0; i < se3_vec.size(); ++i) {
    bool const passed = test(se3_vec[i], se3_vec[(i + 3) % se3_vec.size()]);
    if (!passed) {
      std::cerr << "failed!" << std::endl << std::endl;
      exit(-1);
    }
  }

  Eigen::Matrix<ceres::Jet<double, 28>, 4, 4> mat;
  mat.setIdentity();
  std::cout << CreateSE3FromMatrix(mat) << std::endl;

  return 0;
}
