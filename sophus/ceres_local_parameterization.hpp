#ifndef SOPHUS_CERES_LOCAL_PARAMETERIZATION_HPP
#define SOPHUS_CERES_LOCAL_PARAMETERIZATION_HPP

namespace Sophus {

template <template <typename, int = 0> class LieGroup>
class LocalParameterization : public ceres::LocalParameterization {
 public:
  using LieGroupd = LieGroup<double>;
  using Tangent = typename LieGroupd::Tangent;
  static int constexpr DoF = LieGroupd::DoF;
  static int constexpr num_parameters = LieGroupd::num_parameters;
  virtual ~LocalParameterization() {}

  // LieGroup plus operation for Ceres
  //
  //  T * exp(x)
  //
  virtual bool Plus(double const* T_raw, double const* delta_raw,
                    double* T_plus_delta_raw) const {
    Eigen::Map<LieGroupd const> const T(T_raw);
    Eigen::Map<Tangent const> const delta(delta_raw);
    Eigen::Map<LieGroupd> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * LieGroupd::exp(delta);
    return true;
  }

  // Jacobian of LieGroup plus operation for Ceres
  //
  // Dx T * exp(x)  with  x=0
  //
  virtual bool ComputeJacobian(double const* T_raw,
                               double* jacobian_raw) const {
    Eigen::Map<LieGroupd const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, num_parameters, DoF,
                             DoF == 1 ? Eigen::ColMajor : Eigen::RowMajor>>
        jacobian(jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }

  virtual int GlobalSize() const { return LieGroupd::num_parameters; }

  virtual int LocalSize() const { return LieGroupd::DoF; }
};

}  // namespace Sophus

#endif
