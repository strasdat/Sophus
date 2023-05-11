// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/ceres/manifold.h"
#include "sophus/common/common.h"
#include "sophus/lie/isometry3.h"
#include "sophus/sensor/camera_rig.h"

#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <iostream>

namespace sophus::test {

struct PointTrack {
  std::map<int, Eigen::Vector2d> observations;
};

struct PointTracks {
  std::vector<PointTrack> point_track_for_camera;
};

struct Variables {
  std::vector<sophus::SE3d> world_from_robot_path;
  std::vector<Eigen::Vector3d> points_in_world;
};

struct MiniSim {
  static int constexpr kNumPoses = 5;
  static int constexpr kNumPoints = 50;

  MiniSim() {
    int width = 640;
    int height = 480;
    sophus::CameraModel pinhole_intrinsics =
        sophus::CameraModel::createDefaultPinholeModel({width, height});

    CameraInRig cam_right(pinhole_intrinsics);
    CameraInRig cam_left = cam_right;
    cam_right.rig_from_camera = sophus::Isometry3F64::fromTy(0.25);
    cam_left.rig_from_camera = sophus::Isometry3F64::fromTy(-0.25);
    camera_rig.cameras_in_rig.push_back(cam_right);
    camera_rig.cameras_in_rig.push_back(cam_left);

    for (int i = 0; i < kNumPoses; ++i) {
      truth.world_from_robot_path.push_back(
          sophus::Isometry3F64::fromTx(0.1 * i));
    }

    sophus::Isometry3F64 world_from_robot_final_pose =
        truth.world_from_robot_path.back();

    // uniform random distribution between 0 and 1
    std::default_random_engine re;
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 0.25);

    for (int i = 0; i < kNumPoints; ++i) {
      Eigen::Vector2d pixel =
          Eigen::Vector2d(unif(re) * width, unif(re) * height);
      double z = unif(re) * 5.0 + 1.0;
      Eigen::Vector3d point_in_right_camera =
          cam_right.camera_model.camUnproj(pixel, z);
      Eigen::Vector3d point_in_left_camera =
          cam_left.rig_from_camera * cam_right.rig_from_camera.inverse() *
          point_in_right_camera;
      if (point_in_left_camera.z() < 0.0) {
        continue;
      }
      Eigen::Vector2d pixel_in_left =
          cam_left.camera_model.camProj(point_in_left_camera);
      if (!cam_left.camera_model.contains(pixel_in_left)) {
        continue;
      }

      truth.points_in_world.push_back(
          world_from_robot_final_pose * point_in_left_camera);
      PointTracks point_tracks;
      point_tracks.point_track_for_camera.resize(2);
      observations.push_back(point_tracks);
    }

    for (size_t point_idx = 0; point_idx < truth.points_in_world.size();
         ++point_idx) {
      for (size_t pose_idx = 0; pose_idx < truth.world_from_robot_path.size();
           ++pose_idx) {
        sophus::Isometry3F64 const& world_from_robot =
            truth.world_from_robot_path[pose_idx];
        PointTracks& point_track = SOPHUS_AT(observations, point_idx);
        Eigen::Vector3d point_in_world = truth.points_in_world[point_idx];
        Eigen::Vector3d point_in_robot =
            world_from_robot.inverse() * point_in_world;
        for (size_t cam_idx = 0; cam_idx < camera_rig.cameras_in_rig.size();
             ++cam_idx) {
          CameraInRig const& camera_in_rig = camera_rig.cameras_in_rig[cam_idx];
          Eigen::Vector3d point_in_camera =
              camera_in_rig.rig_from_camera.inverse() * point_in_robot;
          Eigen::Vector2d pixel =
              camera_in_rig.camera_model.camProj(point_in_camera);
          pixel.x() += normal(re);
          pixel.y() += normal(re);

          if (camera_in_rig.camera_model.contains(pixel)) {
            SOPHUS_AT(point_track.point_track_for_camera, cam_idx)
                .observations.insert(std::make_pair(pose_idx, pixel));
          }
        }
      }
    }
  }

  Variables truth;
  MultiCameraRig camera_rig;
  std::vector<PointTracks> observations;
};

struct Cost {
  double median() const {
    SOPHUS_ASSERT(!cost_terms.empty());
    std::vector<double> sorted = cost_terms;
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() / 2];
  }

  double mean() const {
    SOPHUS_ASSERT(!cost_terms.empty());
    double sum = 0.0;
    for (double cost : cost_terms) {
      sum += cost;
    }
    return sum / cost_terms.size();
  }

  std::vector<double> cost_terms;
};

Cost cost(
    std::vector<PointTracks> const& observations,
    MultiCameraRig const& camera_rig,
    Variables const& estimate) {
  Cost cost;
  for (size_t point_idx = 0; point_idx < estimate.points_in_world.size();
       ++point_idx) {
    PointTracks const& point_track = SOPHUS_AT(observations, point_idx);
    for (size_t cam_idx = 0; cam_idx < camera_rig.cameras_in_rig.size();
         ++cam_idx) {
      CameraInRig const& camera_in_rig =
          SOPHUS_AT(camera_rig.cameras_in_rig, cam_idx);

      std::map<int, Eigen::Vector2d> const& obs =
          point_track.point_track_for_camera[cam_idx].observations;

      for (auto const& [pose_id, pixel] : obs) {
        sophus::Isometry3F64 const& world_from_robot =
            SOPHUS_AT(estimate.world_from_robot_path, pose_id);
        Eigen::Vector3d point_in_robot =
            world_from_robot.inverse() * estimate.points_in_world[point_idx];
        Eigen::Vector3d point_in_camera =
            camera_in_rig.rig_from_camera.inverse() * point_in_robot;
        Eigen::Vector2d pixel_reproj =
            camera_in_rig.camera_model.camProj(point_in_camera);
        cost.cost_terms.push_back((pixel - pixel_reproj).norm());
      }
    }
  }
  return cost;
}

struct BundleAdjustCostFunctor {
  BundleAdjustCostFunctor(
      sophus::PinholeModel const& camera_model,
      sophus::Isometry3F64 const& robot_from_camera,
      Eigen::Vector2d const& observation)
      : camera_model(camera_model),
        robot_from_camera(robot_from_camera),
        observation(observation) {}

  template <typename T>
  bool operator()(
      T const* const world_from_robot_raw,
      T const* const point_in_world_raw,
      T* residuals_raw) const {
    sophus::Isometry3<T> world_from_robot = sophus::Isometry3<T>::fromParams(
        Eigen::Map<Eigen::Matrix<T, 7, 1> const>(world_from_robot_raw));
    Eigen::Matrix<T, 3, 1> point_in_world =
        Eigen::Map<Eigen::Matrix<T, 3, 1> const>(point_in_world_raw);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_raw);

    Eigen::Matrix<T, 3, 1> point_in_camera =
        (world_from_robot * robot_from_camera.cast<T>()).inverse() *
        point_in_world;

    Eigen::Matrix<T, 2, 1> pixel =
        camera_model.cast<T>().camProj(point_in_camera);

    residuals = pixel - observation.cast<T>();
    return true;
  }

  sophus::PinholeModel camera_model;
  sophus::Isometry3F64 robot_from_camera;
  Eigen::Vector2d observation;
};

void ceres_optimization(MiniSim const& sim, Variables& estimate) {
  ::ceres::Problem problem;

  auto parametrization = new sophus::ceres::Manifold<sophus::Isometry3>;

  for (size_t pose_idx = 0; pose_idx < estimate.world_from_robot_path.size();
       ++pose_idx) {
    sophus::Isometry3F64& world_from_robot =
        estimate.world_from_robot_path[pose_idx];
    problem.AddParameterBlock(
        world_from_robot.unsafeMutPtr(),
        sophus::Isometry3F64::kNumParams,
        parametrization);

    if (pose_idx == 0) {
      problem.SetParameterBlockConstant(world_from_robot.unsafeMutPtr());
    }
  }

  for (size_t point_idx = 0; point_idx < estimate.points_in_world.size();
       ++point_idx) {
    PointTracks const& point_track = SOPHUS_AT(sim.observations, point_idx);
    for (size_t cam_idx = 0; cam_idx < sim.camera_rig.cameras_in_rig.size();
         ++cam_idx) {
      CameraInRig const& camera_in_rig =
          SOPHUS_AT(sim.camera_rig.cameras_in_rig, cam_idx);
      std::map<int, Eigen::Vector2d> const& obs =
          point_track.point_track_for_camera[cam_idx].observations;

      for (auto const& [pose_id, pixel] : obs) {
        sophus::Isometry3F64& world_from_robot =
            estimate.world_from_robot_path[pose_id];

        ::ceres::CostFunction* cost_function =
            new ::ceres::AutoDiffCostFunction<BundleAdjustCostFunctor, 2, 7, 3>(
                new BundleAdjustCostFunctor(
                    std::get<sophus::PinholeModel>(
                        camera_in_rig.camera_model.modelVariant()),
                    camera_in_rig.rig_from_camera,
                    pixel));

        problem.AddResidualBlock(
            cost_function,
            nullptr,
            estimate.world_from_robot_path[pose_id].unsafeMutPtr(),
            estimate.points_in_world[point_idx].data());
      }
    }
  }

  ::ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ::ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  ::ceres::Solver::Summary summary;
  ::ceres::Solve(options, &problem, &summary);
  SOPHUS_INFO("Ceres summary: {}", summary.BriefReport());
}

TEST(bundle_adjust, test) {
  MiniSim sim;

  SOPHUS_ASSERT_EQ(sim.truth.world_from_robot_path.size(), MiniSim::kNumPoses);
  SOPHUS_ASSERT_GE(sim.truth.points_in_world.size(), 0.5 * MiniSim::kNumPoints);
  SOPHUS_ASSERT_EQ(sim.truth.points_in_world.size(), sim.observations.size());

  Variables est = sim.truth;

  Cost c = cost(sim.observations, sim.camera_rig, est);
  SOPHUS_INFO("Initial cost: median {}, mean {}", c.median(), c.mean());
  SOPHUS_ASSERT_LE(c.median(), 0.5);
  SOPHUS_ASSERT_LE(c.mean(), 0.5);

  ceres_optimization(sim, est);
  c = cost(sim.observations, sim.camera_rig, est);
  SOPHUS_INFO("Final cost: median {}, mean {}", c.median(), c.mean());
  SOPHUS_ASSERT_LE(c.median(), 0.5);
  SOPHUS_ASSERT_LE(c.mean(), 0.5);

  for (size_t i = 0; i < est.world_from_robot_path.size(); ++i) {
    SOPHUS_ASSERT_NEAR(
        est.world_from_robot_path[i].compactMatrix(),
        sim.truth.world_from_robot_path[i].compactMatrix(),
        0.01,
        "pose {}",
        i);
  }

  est = sim.truth;

  // adding some error to the path
  for (size_t i = 1; i < est.world_from_robot_path.size(); ++i) {
    est.world_from_robot_path[i].translation() +=
        Eigen::Vector3d(0.3, -0.1, 0.1);
    est.world_from_robot_path[i].setRotation(
        est.world_from_robot_path[i].rotation() *
        sophus::Rotation3F64::fromRx(0.1));
  }

  c = cost(sim.observations, sim.camera_rig, est);
  SOPHUS_INFO("Initial cost: median {}, mean {}", c.median(), c.mean());
  SOPHUS_ASSERT_GE(c.median(), 0.5);
  SOPHUS_ASSERT_GE(c.mean(), 0.5);

  ceres_optimization(sim, est);
  c = cost(sim.observations, sim.camera_rig, est);
  SOPHUS_INFO("Final cost: median {}, mean {}", c.median(), c.mean());
  SOPHUS_ASSERT_LE(c.median(), 0.5);
  SOPHUS_ASSERT_LE(c.mean(), 0.5);

  for (size_t i = 0; i < est.world_from_robot_path.size(); ++i) {
    SOPHUS_ASSERT_NEAR(
        est.world_from_robot_path[i].compactMatrix(),
        sim.truth.world_from_robot_path[i].compactMatrix(),
        0.01,
        "pose {}",
        i);
  }
}

}  // namespace sophus::test
