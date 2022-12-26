// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/ceres/typetraits.h"

namespace sophus {

/// Templated local parameterization for LieGroup [with implemented
/// LieGroup::Dx_this_mul_exp_x_at_0() ]
template <template <class, int = 0> class TLieGroup>
class Manifold : public ceres::Manifold {
 public:
  using LieGroupF64 = TLieGroup<double>;
  using Tangent = typename LieGroupF64::Tangent;
  using TangentMap = typename sophus::Mapper<Tangent>::Map;
  using TangentConstMap = typename sophus::Mapper<Tangent>::ConstMap;
  static int constexpr kDoF = LieGroupF64::kDoF;
  static int constexpr kNumParameters = LieGroupF64::kNumParameters;

  /// LieGroup plus operation for Ceres
  ///
  ///  T * exp(x)
  ///
  bool Plus(
      double const* t_raw,
      double const* delta_raw,
      double* t_plus_delta_raw) const override {
    Eigen::Map<LieGroupF64 const> const t(t_raw);
    TangentConstMap delta = sophus::Mapper<Tangent>::map(delta_raw);
    Eigen::Map<LieGroupF64> t_plus_delta(t_plus_delta_raw);
    t_plus_delta = t * LieGroupF64::exp(delta);
    return true;
  }

  /// Jacobian of LieGroup plus operation for Ceres
  ///
  /// Dx T * exp(x)  with  x=0
  ///
  bool PlusJacobian(double const* t_raw, double* jacobian_raw) const override {
    Eigen::Map<LieGroupF64 const> t(t_raw);
    Eigen::Map<Eigen::Matrix<
        double,
        kNumParameters,
        kDoF,
        kDoF == 1 ? Eigen::ColMajor : Eigen::RowMajor>>
        jacobian(jacobian_raw);
    jacobian = t.dxThisMulExpXAt0();
    return true;
  }

  bool Minus(double const* y_raw, double const* x_raw, double* y_minus_x_raw)
      const override {
    Eigen::Map<LieGroupF64 const> y(y_raw);

    Eigen::Map<LieGroupF64 const> x(x_raw);
    TangentMap y_minus_x = sophus::Mapper<Tangent>::map(y_minus_x_raw);

    y_minus_x = (x.inverse() * y).log();
    return true;
  }

  bool MinusJacobian(double const* x_raw, double* jacobian_raw) const override {
    Eigen::Map<LieGroupF64 const> x(x_raw);
    Eigen::Map<Eigen::Matrix<double, kDoF, kNumParameters, Eigen::RowMajor>>
        jacobian(jacobian_raw);
    jacobian = x.dxLogThisInvTimesXAtThis();
    return true;
  }

  [[nodiscard]] int AmbientSize() const override {
    return LieGroupF64::kNumParameters;
  }

  [[nodiscard]] int TangentSize() const override { return LieGroupF64::kDoF; }
};

}  // namespace sophus
