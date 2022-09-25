// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/experimental/least_squares.h"

#include "sophus/lie/se3.h"
#include "sophus/sensor/camera_model.h"

#include <farm_ng/core/logging/logger.h>
#include <gtest/gtest.h>

#include <optional>

using namespace sophus;
using namespace sophus::experimental;

/// Example of a binary cost function
///
/// Reprojection error:
///  - Arg 0: SE(3) pose, tangent dim 6
///  - Arg 1: 3-point,    tangent dim 3
///
/// Residual: pixel reprojection 2-vector
class PosePointReprojFunctor {
 public:
  PosePointReprojFunctor(PinholeModel camera) : camera(camera) {}

  static constexpr std::array<int, 2> kArgsDimArray = {{6, 3}};

  using ConstantType = Eigen::Vector2d;

  template <class TArgTypes>
  [[nodiscard]] std::optional<LeastSquaresCostTermState<TArgTypes::kBlockDim>>
  evalCostTerm(
      sophus::Se3F64 const& camera_pose_world,
      Eigen::Vector<double, kArgsDimArray[1]> point_in_world,
      ConstantType const& obs) const {
    static int constexpr kBlockDim = TArgTypes::kBlockDim;

    Eigen::Vector3d point_in_camera = camera_pose_world * point_in_world;
    Eigen::Vector2d pixel = camera.camProj(point_in_camera);
    Eigen::Vector2d residual = pixel - obs;

    LeastSquaresCostTermState<kBlockDim> state;
    state.cost = residual.dot(residual);

    if constexpr (kBlockDim == 0) {
      return state;
    } else {
      Eigen::Matrix<double, 2, kBlockDim> dx_residual;

      [[maybe_unused]] auto eval_pose_jac = [&] {
        return camera.dxCamProjExpXPointAt0(point_in_camera);
      };
      [[maybe_unused]] auto eval_point_jac = [&] {
        return camera.dxCamProjX(point_in_camera);
      };

      // TODO(begin): imp generic constexpr or function template for this
      if constexpr (
          TArgTypes::kArgTypeArray[0] == ArgType::variable &&
          TArgTypes::kArgTypeArray[1] == ArgType::variable) {
        static_assert(kBlockDim == 9);
        dx_residual.template leftCols<6>() = eval_pose_jac();
        dx_residual.template rightCols<3>() = eval_point_jac();
      } else {
        if constexpr (TArgTypes::kArgTypeArray[1] == ArgType::variable) {
          static_assert(kBlockDim == 3);
          dx_residual = eval_point_jac();
        } else {
          if constexpr (TArgTypes::kArgTypeArray[0] == ArgType::variable) {
            static_assert(kBlockDim == 6);
            dx_residual = eval_pose_jac();
          } else {
            static_assert(kBlockDim == 0);
            return state;
          }
        }
      }
      // TODO(end)

      state.hessian_block = dx_residual.transpose() * dx_residual;
      state.gradient_segment = dx_residual.transpose() * residual;

      return state;
    }
  }

  CameraModel camera;
};

TEST(binary_cost, smoke) {
  PinholeModel cam = createDefaultPinholeModel({64, 40});

  ManifoldFamily<6, Se3F64> pose_family;
  pose_family.manifolds.push_back(Se3F64::transX(0.01));
  pose_family.manifolds.push_back(Se3F64::transY(0.02));

  ManifoldFamily<3> point_family;
  point_family.manifolds.push_back(Eigen::Vector3d(0, 0, 10));
  point_family.manifolds.push_back(Eigen::Vector3d(-1, 0, 10));
  point_family.manifolds.push_back(Eigen::Vector3d(0, 2, 10));
  point_family.manifolds.push_back(Eigen::Vector3d(1, 2.5, 10));

  // vars: poses, points; fixed: n/a
  {
    using ArgsList = CostTermRef<2, Eigen::Vector2d>;
    std::vector<ArgsList> cost_terms;
    for (size_t point_id = 0; point_id < point_family.manifolds.size();
         ++point_id) {
      ArgsList arg_list;
      arg_list.arg_ids[1] = point_id;
      std::vector<Eigen::Vector2d> point_track;
      for (size_t pose_id = 0; pose_id < pose_family.manifolds.size();
           ++pose_id) {
        arg_list.arg_ids[0] = pose_id;
        Eigen::Vector3d point_in_image =
            pose_family.manifolds[pose_id] * point_family.manifolds[point_id];
        Eigen::Vector2d pixel = cam.camProj(point_in_image);
        arg_list.constant = pixel + Eigen::Vector2d(0.5, 0);
        cost_terms.push_back(arg_list);
      }
    }

    PosePointReprojFunctor ba2(cam);
    static_assert(PosePointReprojFunctor::kArgsDimArray.size() == 2);
    auto family_state =
        apply(ba2, cost_terms, Var(pose_family), Var(point_family));

    using A = ArgTypes<
        false,
        PosePointReprojFunctor,
        decltype(Var(pose_family)),
        decltype(Var(pose_family))>;

    static_assert(A::kNumArgs == 2);
    static_assert(A::kNumVarArgs == 2);
    static_assert(A::kBlockDim == 0);

    auto family_state2 =
        apply<false>(ba2, cost_terms, Var(pose_family), Var(point_family));

    // vars: poses; fixed: points
    { apply(ba2, cost_terms, Var(pose_family), CondVar(point_family)); }

    // fixed: poses; var: points
    { apply(ba2, cost_terms, Var(pose_family), CondVar(point_family)); }
  }
}

class CompileOnlyTernaryCostFunctorExample {
 public:
  CompileOnlyTernaryCostFunctorExample() {}

  using ConstantType = farm_ng::Void;

  static constexpr std::array<int, 3> kArgsDimArray = {{4, 6, 3}};

  template <class TArgTypes>
  [[nodiscard]] std::optional<LeastSquaresCostTermState<TArgTypes::kBlockDim>>
  evalCostTerm(
      Eigen::Vector<double, kArgsDimArray[0]> const& /*unused*/,
      sophus::Se3F64 const& /*unused*/,
      Eigen::Vector<double, kArgsDimArray[2]> const& /*unused*/,
      ConstantType /*unused*/) const {
    LeastSquaresCostTermState<TArgTypes::kBlockDim> state;
    return state;
  }
};

TEST(ternary_cost, compile_test) {
  ManifoldFamily<4> cam_family;
  ManifoldFamily<6, Se3F64> pose_family;
  ManifoldFamily<3> point_family;

  CompileOnlyTernaryCostFunctorExample ba;
  std::vector<CostTermRef<3>> arg_ids;

  {
    apply(ba, arg_ids, Var(cam_family), Var(pose_family), Var(point_family));

    apply<false>(
        ba, arg_ids, Var(cam_family), Var(pose_family), Var(point_family));
  }

  {
    apply(
        ba, arg_ids, CondVar(cam_family), Var(pose_family), Var(point_family));
  }

  {
    apply(
        ba, arg_ids, Var(cam_family), CondVar(pose_family), Var(point_family));
  }

  {
    apply(
        ba, arg_ids, Var(cam_family), Var(pose_family), CondVar(point_family));
  }

  {
    apply(
        ba,
        arg_ids,
        CondVar(cam_family),
        CondVar(pose_family),
        Var(point_family));
  }

  {
    apply(
        ba,
        arg_ids,
        Var(cam_family),
        CondVar(pose_family),
        CondVar(point_family));
  }

  {
    apply(
        ba,
        arg_ids,
        CondVar(cam_family),
        Var(pose_family),
        CondVar(point_family));
  }
}
