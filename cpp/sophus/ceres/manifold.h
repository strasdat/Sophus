// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/ceres/typetraits.h"

namespace sophus::ceres {

/// Templated local parameterization for LieGroup [with implemented
/// LieGroup::Dx_this_mul_exp_x_at_0() ]
template <template <class> class TLieGroup>
class Manifold : public ::ceres::Manifold {
 public:
  using LieGroupF64 = TLieGroup<double>;
  static int constexpr kDof = LieGroupF64::kDof;
  static int constexpr kNumParams = LieGroupF64::kNumParams;

  using Tangent = Eigen::Vector<double, kDof>;
  using Params = Eigen::Vector<double, kNumParams>;

  /// LieGroup plus operation for Ceres
  ///
  ///  T * exp(x)
  ///
  bool Plus(
      double const* t_raw,
      double const* delta_raw,
      double* t_plus_delta_raw) const override {
    LieGroupF64 t = LieGroupF64::fromParams(Eigen::Map<Params const>(t_raw));
    Eigen::Map<Tangent const> delta(delta_raw);

    Eigen::Map<Params> out_params(t_plus_delta_raw);
    LieGroupF64 t_plus_delta = t * LieGroupF64::exp(delta);
    out_params = t_plus_delta.params();
    return true;
  }

  /// Jacobian of LieGroup plus operation for Ceres
  ///
  /// Dx T * exp(x)  with  x=0
  ///
  bool PlusJacobian(double const* t_raw, double* jacobian_raw) const override {
    LieGroupF64 t = LieGroupF64::fromParams(Eigen::Map<Params const>(t_raw));
    Eigen::Map<Eigen::Matrix<
        double,
        kNumParams,
        kDof,
        kDof == 1 ? Eigen::ColMajor : Eigen::RowMajor>>
        jacobian(jacobian_raw);
    jacobian = t.dxThisMulExpXAt0();
    return true;
  }

  bool Minus(double const* y_raw, double const* x_raw, double* y_minus_x_raw)
      const override {
    LieGroupF64 y = LieGroupF64::fromParams(Eigen::Map<Params const>(y_raw));
    LieGroupF64 x = LieGroupF64::fromParams(Eigen::Map<Params const>(x_raw));
    Eigen::Map<Tangent> out_params(y_minus_x_raw);
    out_params = (x.inverse() * y).log();
    return true;
  }

  bool MinusJacobian(double const* x_raw, double* jacobian_raw) const override {
    LieGroupF64 x = LieGroupF64::fromParams(Eigen::Map<Params const>(x_raw));
    Eigen::Map<Eigen::Matrix<double, kDof, kNumParams, Eigen::RowMajor>>
        jacobian(jacobian_raw);
    jacobian = x.dxLogThisInvTimesXAtThis();
    return true;
  }

  [[nodiscard]] int AmbientSize() const override {
    return LieGroupF64::kNumParams;
  }

  [[nodiscard]] int TangentSize() const override { return LieGroupF64::kDof; }
};

}  // namespace sophus::ceres
