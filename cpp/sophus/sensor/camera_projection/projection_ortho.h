// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include <Eigen/Core>

namespace sophus {

struct ProjectionOrtho {
  template <class TPoints, int kRows>
  using WithRows = Eigen::
      Matrix<typename TPoints::Scalar, kRows, TPoints::ColsAtCompileTime>;

  // Project one or more 3-points from the camera frame into the canonical z=0
  // plane through orthographic projection. For N points, a 3xN matrix must be
  // provided where each column is a point to be transformed. The result will be
  // a 2xN matrix. N may be dynamically sized, but the input columns must be
  // statically determined as 3 at compile time.
  template <class TDerived>
  static auto proj(Eigen::DenseBase<TDerived> const& points_in_camera)
      -> WithRows<Eigen::DenseBase<TDerived>, 2> {
    static_assert(TDerived::RowsAtCompileTime == 3);
    return points_in_camera.template topRows<2>();
  }

  template <class TDerived>
  static auto unproj(
      Eigen::DenseBase<TDerived> const& points_in_cam_canonical,
      typename TDerived::Scalar extension =
          static_cast<typename TDerived::Scalar>(0.0))
      -> WithRows<Eigen::DenseBase<TDerived>, 3> {
    static_assert(TDerived::RowsAtCompileTime == 2);
    WithRows<Eigen::DenseBase<TDerived>, 3> unprojected;
    unprojected.template topRows<2>() = points_in_cam_canonical;
    unprojected.template bottomRows<1>() =
        WithRows<Eigen::DenseBase<TDerived>, 1>::Constant(
            points_in_cam_canonical.cols(), extension);
    return unprojected;
  }

  /// Returns point derivative of inverse depth point projection:
  ///
  ///   Dx proj(x) with x = (a,b,psi) being an inverse depth point.
  template <class TScalar>
  static auto dxProjX(Eigen::Matrix<TScalar, 3, 1> const& /*unused*/)
      -> Eigen::Matrix<TScalar, 2, 3> {
    SOPHUS_UNIMPLEMENTED();
  }
};

}  // namespace sophus
