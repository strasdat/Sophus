#pragma once

#include <ceres/manifold.h>
#include <sophus/ceres_typetraits.hpp>

namespace Sophus {

/// Templated local parameterization for LieGroup [with implemented
/// LieGroup::Dx_this_mul_exp_x_at_0() ]
template <template <typename, int = 0> class LieGroup>
class Manifold : public ceres::Manifold {
 public:
  using LieGroupd = LieGroup<double>;
  using Tangent = typename LieGroupd::Tangent;
  using TangentMap = typename Sophus::Mapper<Tangent>::Map;
  using TangentConstMap = typename Sophus::Mapper<Tangent>::ConstMap;
  static int constexpr DoF = LieGroupd::DoF;
  static int constexpr num_parameters = LieGroupd::num_parameters;

  /// LieGroup plus operation for Ceres
  ///
  ///  T * exp(x)
  ///
  bool Plus(double const* T_raw, double const* delta_raw,
            double* T_plus_delta_raw) const override {
    Eigen::Map<LieGroupd const> const T(T_raw);
    TangentConstMap delta = Sophus::Mapper<Tangent>::map(delta_raw);
    Eigen::Map<LieGroupd> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * LieGroupd::exp(delta);
    return true;
  }

  /// Jacobian of LieGroup plus operation for Ceres
  ///
  /// Dx T * exp(x)  with  x=0
  ///
  bool PlusJacobian(double const* T_raw, double* jacobian_raw) const override {
    Eigen::Map<LieGroupd const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, num_parameters, DoF,
                             DoF == 1 ? Eigen::ColMajor : Eigen::RowMajor>>
        jacobian(jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }

  bool Minus(double const* y_raw, double const* x_raw,
             double* y_minus_x_raw) const override {
    Eigen::Map<LieGroupd const> y(y_raw), x(x_raw);
    TangentMap y_minus_x = Sophus::Mapper<Tangent>::map(y_minus_x_raw);

    y_minus_x = (x.inverse() * y).log();
    return true;
  }

  bool MinusJacobian(double const* x_raw, double* jacobian_raw) const override {
    Eigen::Map<LieGroupd const> x(x_raw);
    Eigen::Map<Eigen::Matrix<double, DoF, num_parameters, Eigen::RowMajor>>
        jacobian(jacobian_raw);
    jacobian = x.Dx_log_this_inv_by_x_at_this();
    return true;
  }

  int AmbientSize() const override { return LieGroupd::num_parameters; }

  int TangentSize() const override { return LieGroupd::DoF; }
};

}  // namespace Sophus
