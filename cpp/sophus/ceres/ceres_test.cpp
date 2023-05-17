// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/ceres/manifold.h"
#include "sophus/common/common.h"
#include "sophus/lie/identity.h"
#include "sophus/lie/isometry2.h"
#include "sophus/lie/isometry3.h"
#include "sophus/lie/scaling.h"
#include "sophus/lie/scaling_translation.h"
#include "sophus/lie/similarity2.h"
#include "sophus/lie/similarity3.h"
#include "sophus/lie/spiral_similarity2.h"
#include "sophus/lie/spiral_similarity3.h"
#include "sophus/lie/translation.h"
#include "sophus/manifold/complex.h"
#include "sophus/manifold/quaternion.h"
#include "sophus/sensor/camera_rig.h"

#include <ceres/ceres.h>
#include <farm_ng/core/pipeline/component.h>
#include <gtest/gtest.h>

#include <iostream>

namespace sophus::test {

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

template <template <class> class TTransform>
struct SimplePriorProblem {
  struct Variables {
    TTransform<double> world_from_robot;
  };

  struct CostFunctor {
    CostFunctor(TTransform<double> world_from_robot_prior)
        : world_from_robot_prior_(world_from_robot_prior) {}

    template <typename T>
    bool operator()(
        T const* const world_from_robot_raw, T* residuals_raw) const {
      TTransform<T> world_from_robot = TTransform<T>::fromParams(
          Eigen::Map<Eigen::Matrix<T, TTransform<T>::kNumParams, 1> const>(
              world_from_robot_raw));
      Eigen::Map<Eigen::Matrix<T, TTransform<T>::kDof, 1>> residuals(
          residuals_raw);

      residuals = (world_from_robot.inverse() * world_from_robot_prior_).log();
      return true;
    }

    TTransform<double> world_from_robot_prior_;
  };

  SimplePriorProblem(TTransform<double> world_from_robot_prior)
      : world_from_robot_prior_(world_from_robot_prior),
        truth_({.world_from_robot = world_from_robot_prior}) {}

  void solve(Variables& estimate) {
    ::ceres::Problem problem;

    auto parametrization = new sophus::ceres::Manifold<TTransform>;

    ::ceres::CostFunction* cost_function = new ::ceres::AutoDiffCostFunction<
        CostFunctor,
        TTransform<double>::kDof,
        TTransform<double>::kNumParams>(
        new CostFunctor(world_from_robot_prior_));

    problem.AddParameterBlock(
        estimate.world_from_robot.unsafeMutPtr(),
        TTransform<double>::kNumParams,
        parametrization);

    problem.AddResidualBlock(
        cost_function, nullptr, estimate.world_from_robot.unsafeMutPtr());

    ::ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ::ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ::ceres::Solver::Summary summary;
    ::ceres::Solve(options, &problem, &summary);
    SOPHUS_INFO("Ceres summary: {}", summary.BriefReport());
  }

  double cost(Variables const& estimate) const {
    return (estimate.world_from_robot.inverse() * world_from_robot_prior_)
        .log()
        .norm();
  }

  void test(
      std::function<void(TTransform<double>&)> perturb,
      std::function<Expected<Success>(
          TTransform<double> const&, TTransform<double> const&)> is_near) {
    std::string type_id = ::farm_ng::demangleTypeid(typeid(TTransform<double>));

    Variables est = this->truth_;
    perturb(est.world_from_robot);

    double c = cost(est);
    SOPHUS_INFO("Initial cost: {}", c);

    this->solve(est);
    c = cost(est);
    SOPHUS_INFO("Final cost: {}", c);
    FARM_ASSERT(
        is_near(est.world_from_robot, this->truth_.world_from_robot),
        "Failed for {}",
        type_id);
  }

  TTransform<double> world_from_robot_prior_;

  Variables truth_;
};

template <template <class> class TTransform, int kDim>
struct TransformGraphProblem {
  struct Variables {
    std::vector<TTransform<double>> world_from_robot_transforms;
  };

  struct Observation {
    int robot1_idx;
    int robot2_idx;
    TTransform<double> robot1_from_robot2;
  };

  struct CostFunctor {
    CostFunctor(TTransform<double> const& robot1_from_robot2)
        : robot1_from_robot2_(robot1_from_robot2) {}

    template <typename T>
    bool operator()(
        T const* const world_from_robot1_raw,
        T const* const world_from_robot2_raw,
        T* residuals_raw) const {
      TTransform<T> world_from_robot1 = TTransform<T>::fromParams(
          Eigen::Map<Eigen::Matrix<T, TTransform<T>::kNumParams, 1> const>(
              world_from_robot1_raw));
      TTransform<T> world_from_robot2 = TTransform<T>::fromParams(
          Eigen::Map<Eigen::Matrix<T, TTransform<T>::kNumParams, 1> const>(
              world_from_robot2_raw));
      Eigen::Map<Eigen::Matrix<T, TTransform<T>::kDof, 1>> residuals(
          residuals_raw);

      residuals = ((world_from_robot2.inverse() * world_from_robot1) *
                   robot1_from_robot2_)
                      .log();
      return true;
    }

    TTransform<double> robot1_from_robot2_;
  };

  TransformGraphProblem(
      std::function<auto(double p, TTransform<double>& trans)->void>
          set_subgroup) {
    std::vector<Eigen::Vector2d> positions = {
        Eigen::Vector2d(0, 0),
        Eigen::Vector2d(1, 0),
        Eigen::Vector2d(1.2, 2.0),
        Eigen::Vector2d(0.2, 1.5),
        Eigen::Vector2d(0.0, 0.5)};
    for (size_t i = 0; i < positions.size(); ++i) {
      Eigen::Vector2d p2d = positions[i];
      TTransform<double> world_from_robot;
      Eigen::Vector<double, kDim> position =
          Eigen::Vector<double, kDim>::Zero();
      position.template head<2>() = p2d;
      world_from_robot.translation() = position;
      set_subgroup(double(i) / double(positions.size()), world_from_robot);
      truth_.world_from_robot_transforms.push_back(world_from_robot);
    }

    for (size_t i = 0; i < positions.size(); ++i) {
      Observation obs;
      obs.robot1_idx = i;
      obs.robot2_idx = (i + 1) % positions.size();
      obs.robot1_from_robot2 =
          truth_.world_from_robot_transforms[obs.robot1_idx].inverse() *
          truth_.world_from_robot_transforms[obs.robot2_idx];
      observations_.push_back(obs);
    }
  }

  void solve(Variables& estimate) {
    ::ceres::Problem problem;

    auto parametrization = new sophus::ceres::Manifold<TTransform>;

    for (size_t i = 0; i < estimate.world_from_robot_transforms.size(); ++i) {
      problem.AddParameterBlock(
          estimate.world_from_robot_transforms[i].unsafeMutPtr(),
          TTransform<double>::kNumParams,
          parametrization);

      if (i == 0) {
        problem.SetParameterBlockConstant(
            estimate.world_from_robot_transforms[i].unsafeMutPtr());
      }
    }

    for (Observation const& obs : observations_) {
      ::ceres::CostFunction* cost_function = new ::ceres::AutoDiffCostFunction<
          CostFunctor,
          TTransform<double>::kDof,
          TTransform<double>::kNumParams,
          TTransform<double>::kNumParams>(
          new CostFunctor(obs.robot1_from_robot2));

      problem.AddResidualBlock(
          cost_function,
          nullptr,
          estimate.world_from_robot_transforms[obs.robot1_idx].unsafeMutPtr(),
          estimate.world_from_robot_transforms[obs.robot2_idx].unsafeMutPtr());
    }

    ::ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ::ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ::ceres::Solver::Summary summary;
    ::ceres::Solve(options, &problem, &summary);
    SOPHUS_INFO("Ceres summary: {}", summary.BriefReport());
  }

  Cost cost(Variables const& estimate) const {
    Cost c;
    for (Observation const& obs : observations_) {
      c.cost_terms.push_back(
          (estimate.world_from_robot_transforms[obs.robot2_idx].inverse() *
           estimate.world_from_robot_transforms[obs.robot1_idx] *
           obs.robot1_from_robot2)
              .log()
              .norm());
    }
    return c;
  }

  void test(
      std::function<void(TTransform<double>&)> perturb,
      std::function<Expected<Success>(
          TTransform<double> const&, TTransform<double> const&)> is_near) {
    std::string type_id = ::farm_ng::demangleTypeid(typeid(TTransform<double>));

    SOPHUS_INFO("- Testing: {}", type_id);

    Variables est = this->truth_;
    for (size_t i = 1; i < est.world_from_robot_transforms.size(); ++i) {
      perturb(est.world_from_robot_transforms[i]);
    }

    Cost c = cost(est);
    SOPHUS_INFO("Initial cost: median {}, mean {}", c.median(), c.mean());

    this->solve(est);
    c = cost(est);
    SOPHUS_INFO("Final  cost: median {}, mean {}", c.median(), c.mean());

    for (size_t i = 0; i < est.world_from_robot_transforms.size(); ++i) {
      SOPHUS_INFO(
          "{}, truth: {}",
          i,
          this->truth_.world_from_robot_transforms[i].matrix());
      SOPHUS_INFO(
          "{}, est: {}", i, est.world_from_robot_transforms[i].matrix());
    }
    for (size_t i = 0; i < est.world_from_robot_transforms.size(); ++i) {
      FARM_ASSERT(
          is_near(
              this->truth_.world_from_robot_transforms[i],
              est.world_from_robot_transforms[i]),
          "Failed for {}",
          type_id);
    }
  }

  std::vector<Observation> observations_;
  Variables truth_;
};

template <template <class> class TTransform>
struct SensorMeasurementProblem {
  struct Variables {
    std::vector<TTransform<double>> world_from_robot_path;
    std::vector<Eigen::Vector3d> points_in_world;
  };

  struct PointTrack {
    std::map<int, Eigen::Vector2d> observations;
  };

  struct PointTracks {
    std::vector<PointTrack> point_track_for_camera;
  };

  struct Sensor {
    Sensor() {}

    /// Camera intrinsics
    PinholeModel intrinsics;

    /// Camera extrinsics
    sophus::Isometry3F64 robot_from_camera;
  };

  struct CostFunctor {
    CostFunctor(
        sophus::PinholeModel const& camera_model,
        Isometry3F64 const& robot_from_camera,
        Eigen::Vector2d const& observation)
        : camera_model(camera_model),
          robot_from_camera(robot_from_camera),
          observation(observation) {}

    template <typename T>
    bool operator()(
        T const* const world_from_robot_raw,
        T const* const point_in_world_raw,
        T* residuals_raw) const {
      TTransform<T> world_from_robot = TTransform<T>::fromParams(
          Eigen::Map<Eigen::Matrix<T, 7, 1> const>(world_from_robot_raw));
      Eigen::Matrix<T, 3, 1> point_in_world =
          Eigen::Map<Eigen::Matrix<T, 3, 1> const>(point_in_world_raw);
      Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_raw);

      Eigen::Matrix<T, 3, 1> point_in_camera =
          world_from_robot.inverse() *
          (robot_from_camera.inverse() * point_in_world);

      Eigen::Matrix<T, 2, 1> pixel =
          camera_model.cast<T>().camProj(point_in_camera);

      residuals = pixel - observation;
      return true;
    }

    sophus::PinholeModel camera_model;
    Isometry3F64 robot_from_camera;
    Eigen::Vector2d observation;
  };
  static int constexpr kNumTransforms = 5;
  static int constexpr kNumPoints = 50;

  SensorMeasurementProblem(
      std::function<std::vector<TTransform<double>>(int)> const& create_path) {
    int width = 640;
    int height = 480;
    sophus::PinholeModel pinhole_intrinsics =
        createDefaultPinholeModel({width, height});

    Sensor cam_right;
    cam_right.intrinsics = pinhole_intrinsics;
    cam_right.robot_from_camera = sophus::Isometry3F64::fromTy(0.25);

    Sensor cam_left;
    cam_left.intrinsics = pinhole_intrinsics;
    cam_left.robot_from_camera = sophus::Isometry3F64::fromTy(-0.25);

    this->camera_rig.push_back(cam_right);
    this->camera_rig.push_back(cam_left);

    truth.world_from_robot_path = create_path(kNumTransforms);

    TTransform<double> world_from_robot_final_transform =
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
          cam_right.intrinsics.camUnproj(pixel, z);
      Eigen::Vector3d point_in_left_camera =
          cam_left.robot_from_camera * cam_right.robot_from_camera.inverse() *
          point_in_right_camera;
      if (point_in_left_camera.z() < 0.0) {
        continue;
      }
      Eigen::Vector2d pixel_in_left =
          cam_left.intrinsics.camProj(point_in_left_camera);
      if (!cam_left.intrinsics.contains(pixel_in_left)) {
        continue;
      }

      truth.points_in_world.push_back(
          world_from_robot_final_transform * point_in_left_camera);
      PointTracks point_tracks;
      point_tracks.point_track_for_camera.resize(2);
      observations.push_back(point_tracks);
    }

    for (size_t point_idx = 0; point_idx < truth.points_in_world.size();
         ++point_idx) {
      for (size_t pose_idx = 0; pose_idx < truth.world_from_robot_path.size();
           ++pose_idx) {
        TTransform<double> const& world_from_robot =
            truth.world_from_robot_path[pose_idx];
        PointTracks& point_track = SOPHUS_AT(observations, point_idx);
        Eigen::Vector3d point_in_world = truth.points_in_world[point_idx];
        Eigen::Vector3d point_in_robot =
            world_from_robot.inverse() * point_in_world;
        for (size_t cam_idx = 0; cam_idx < camera_rig.size(); ++cam_idx) {
          Sensor const& camera_in_rig = camera_rig[cam_idx];
          Eigen::Vector3d point_in_camera =
              camera_in_rig.robot_from_camera.inverse() * point_in_robot;
          Eigen::Vector2d pixel =
              camera_in_rig.intrinsics.camProj(point_in_camera);
          pixel.x() += normal(re);
          pixel.y() += normal(re);

          if (camera_in_rig.intrinsics.contains(pixel)) {
            SOPHUS_AT(point_track.point_track_for_camera, cam_idx)
                .observations.insert(std::make_pair(pose_idx, pixel));
          }
        }
      }
    }
  }

  void solve(Variables& estimate) {
    ::ceres::Problem problem;

    auto parametrization = new sophus::ceres::Manifold<TTransform>;

    for (size_t pose_idx = 0; pose_idx < estimate.world_from_robot_path.size();
         ++pose_idx) {
      TTransform<double>& world_from_robot =
          estimate.world_from_robot_path[pose_idx];
      problem.AddParameterBlock(
          world_from_robot.unsafeMutPtr(),
          TTransform<double>::kNumParams,
          parametrization);

      if (pose_idx == 0) {
        problem.SetParameterBlockConstant(world_from_robot.unsafeMutPtr());
      }
    }

    for (size_t point_idx = 0; point_idx < estimate.points_in_world.size();
         ++point_idx) {
      PointTracks const& point_track = SOPHUS_AT(this->observations, point_idx);
      for (size_t cam_idx = 0; cam_idx < this->camera_rig.size(); ++cam_idx) {
        Sensor const& camera_in_rig = SOPHUS_AT(this->camera_rig, cam_idx);
        std::map<int, Eigen::Vector2d> const& obs =
            point_track.point_track_for_camera[cam_idx].observations;

        for (auto const& [pose_id, pixel] : obs) {
          TTransform<double>& world_from_robot =
              estimate.world_from_robot_path[pose_id];

          ::ceres::CostFunction* cost_function =
              new ::ceres::AutoDiffCostFunction<
                  CostFunctor,
                  2,
                  TTransform<double>::kNumParams,
                  TTransform<double>::kPointDim>(new CostFunctor(
                  camera_in_rig.intrinsics,
                  camera_in_rig.robot_from_camera,
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

  Cost cost(Variables const& estimate) {
    Cost cost;
    for (size_t point_idx = 0; point_idx < estimate.points_in_world.size();
         ++point_idx) {
      PointTracks const& point_track = SOPHUS_AT(observations, point_idx);
      for (size_t cam_idx = 0; cam_idx < camera_rig.size(); ++cam_idx) {
        Sensor const& camera_in_rig = SOPHUS_AT(camera_rig, cam_idx);

        std::map<int, Eigen::Vector2d> const& obs =
            point_track.point_track_for_camera[cam_idx].observations;

        for (auto const& [pose_id, pixel] : obs) {
          TTransform<double> const& world_from_robot =
              SOPHUS_AT(estimate.world_from_robot_path, pose_id);
          Eigen::Vector3d point_in_robot =
              world_from_robot.inverse() * estimate.points_in_world[point_idx];
          Eigen::Vector3d point_in_camera =
              camera_in_rig.robot_from_camera.inverse() * point_in_robot;
          Eigen::Vector2d pixel_reproj =
              camera_in_rig.intrinsics.camProj(point_in_camera);
          cost.cost_terms.push_back((pixel - pixel_reproj).norm());
        }
      }
    }
    return cost;
  }

  void test(
      std::function<void(TTransform<double>&)> perturb,
      std::function<Expected<Success>(
          TTransform<double> const&, TTransform<double> const&)> is_near) {
    SOPHUS_ASSERT_EQ(truth.world_from_robot_path.size(), kNumTransforms);
    SOPHUS_ASSERT_GE(truth.points_in_world.size(), 0.5 * kNumPoints);
    SOPHUS_ASSERT_EQ(truth.points_in_world.size(), observations.size());

    Variables est = truth;

    // adding some error to the path
    for (size_t i = 1; i < est.world_from_robot_path.size(); ++i) {
      perturb(est.world_from_robot_path[i]);
    }

    Cost c = cost(est);
    SOPHUS_INFO("Initial cost: median {}, mean {}", c.median(), c.mean());

    solve(est);
    c = cost(est);
    SOPHUS_INFO("Final cost: median {}, mean {}", c.median(), c.mean());
    SOPHUS_ASSERT_LE(c.median(), 0.5);
    SOPHUS_ASSERT_LE(c.mean(), 0.5);

    for (size_t i = 0; i < est.world_from_robot_path.size(); ++i) {
      SOPHUS_ASSERT(is_near(
          est.world_from_robot_path[i], truth.world_from_robot_path[i]));
    }
  }

  Variables truth;
  std::vector<Sensor> camera_rig;
  std::vector<PointTracks> observations;
};

TEST(ceres_solve, regression_test) {
  static double const kPosThr = 0.002;
  static double const kRotThr = 0.001;
  static double const kScaleThr = 0.005;
  {
    auto perturb = [](Isometry2F64& est) {
      est.translation() += Eigen::Vector2d(0.2, -0.1);
      est.setRotation(est.rotation() * sophus::Rotation2F64(0.1));
    };
    auto is_near = [](Isometry2F64 const& truth,
                      Isometry2F64 const& est) -> Expected<Success> {
      double translation_error =
          (truth.translation() - est.translation()).norm();
      double rotation_error =
          (truth.rotation() * est.rotation().inverse()).log().norm();
      if (translation_error > kPosThr || rotation_error > kRotThr) {
        return FARM_UNEXPECTED(
            "translation error {} [{}], rotation error {} [{}]",
            translation_error,
            kPosThr,
            rotation_error,
            kRotThr);
      }
      return Success{};
    };

    SimplePriorProblem<Isometry2> prior_problem(Isometry2F64{});
    prior_problem.test([](Isometry2F64& est) {}, is_near);
    prior_problem.test(perturb, is_near);

    TransformGraphProblem<Isometry2, 2> graph_problem(
        [](double p, Isometry2F64& est) {});
    graph_problem.test([](Isometry2F64& est) {}, is_near);
    graph_problem.test(perturb, is_near);
  }

  {
    auto perturb = [](Similarity2F64& est) {
      est.translation() += Eigen::Vector2d(0.2, -0.1);
      est.setRotation(est.rotation() * sophus::Rotation2F64(0.1));
      est.setScale(est.scale() * 2.0);
    };
    auto is_near = [](Similarity2F64 const& truth,
                      Similarity2F64 const& est) -> Expected<Success> {
      double translation_error =
          (truth.translation() - est.translation()).norm();
      double rotation_error =
          (truth.rotation() * est.rotation().inverse()).log().norm();
      double scale_error = std::abs(truth.scale() - est.scale());
      if (translation_error > kPosThr || rotation_error > kRotThr ||
          scale_error > kScaleThr) {
        return FARM_UNEXPECTED(
            "translation error {} [{}], rotation error {} [{}], scale error "
            "{} "
            "[{}]",
            translation_error,
            kPosThr,
            rotation_error,
            kRotThr,
            scale_error,
            kScaleThr);
      }
      return Success{};
    };
    SimplePriorProblem<Similarity2> prior_problem(Similarity2F64{});
    prior_problem.test([](Similarity2F64& est) {}, is_near);
    prior_problem.test(perturb, is_near);
    TransformGraphProblem<Similarity2, 2> graph_problem(
        [](double p, Similarity2F64& est) {});
    graph_problem.test([](Similarity2F64& est) {}, is_near);
    graph_problem.test(perturb, is_near);
  }

  {
    auto perturb = [](Translation2F64& est) {
      est.translation() += Eigen::Vector2d(0.2, -0.1);
    };
    auto is_near = [](Translation2F64 const& truth,
                      Translation2F64 const& est) -> Expected<Success> {
      double translation_error =
          (truth.translation() - est.translation()).norm();
      if (translation_error > kPosThr) {
        return FARM_UNEXPECTED(
            "translation error {} [{}]", translation_error, kPosThr);
      }
      return Success{};
    };
    SimplePriorProblem<Translation2> prior_problem(Translation2F64{});
    prior_problem.test([](Translation2F64& est) {}, is_near);
    prior_problem.test(perturb, is_near);
    TransformGraphProblem<Translation2, 2> graph_problem(
        [](double p, Translation2F64& est) {});
    graph_problem.test([](Translation2F64& est) {}, is_near);
    graph_problem.test(perturb, is_near);
  }

  {
    auto perturb = [](ScalingTranslation2F64& est) {
      est.translation() += Eigen::Vector2d(0.2, -0.1);
      est.setScaleFactors(est.scaleFactors() * 2.0);
    };
    auto is_near = [](ScalingTranslation2F64 const& truth,
                      ScalingTranslation2F64 const& est) -> Expected<Success> {
      double translation_error =
          (truth.translation() - est.translation()).norm();
      double scale_error = (truth.scaleFactors() - est.scaleFactors()).norm();
      if (translation_error > kPosThr || scale_error > kScaleThr) {
        return FARM_UNEXPECTED(
            "translation error {} [{}], scale error {} [{}]",
            translation_error,
            kPosThr,
            scale_error,
            kScaleThr);
      }
      return Success{};
    };
    SimplePriorProblem<ScalingTranslation2> prior_problem(
        ScalingTranslation2F64{});
    prior_problem.test([](ScalingTranslation2F64& est) {}, is_near);
    prior_problem.test(perturb, is_near);
    TransformGraphProblem<ScalingTranslation2, 2> graph_problem(
        [](double p, ScalingTranslation2F64& est) {});
    graph_problem.test([](ScalingTranslation2F64& est) {}, is_near);
    graph_problem.test(perturb, is_near);
  }

  {
    auto perturb = [](Isometry3F64& est) {
      est.translation() += Eigen::Vector3d(0.3, -0.1, 0.1);
      est.setRotation(est.rotation() * sophus::Rotation3F64::fromRx(0.1));
    };
    auto is_near = [](Isometry3F64 const& truth,
                      Isometry3F64 const& est) -> Expected<Success> {
      double translation_error =
          (truth.translation() - est.translation()).norm();
      double rotation_error =
          (truth.rotation() * est.rotation().inverse()).log().norm();
      if (translation_error > kPosThr || rotation_error > kRotThr) {
        return FARM_UNEXPECTED(
            "translation error {} [{}], rotation error {} [{}]",
            translation_error,
            kPosThr,
            rotation_error,
            kRotThr);
      }
      return Success{};
    };
    SimplePriorProblem<Isometry3> prior_problem(Isometry3F64{});
    prior_problem.test([](Isometry3F64& est) {}, is_near);
    prior_problem.test(perturb, is_near);
    TransformGraphProblem<Isometry3, 3> graph_problem(
        [](double p, Isometry3F64& est) {});
    graph_problem.test([](Isometry3F64& est) {}, is_near);
    graph_problem.test(perturb, is_near);

    SensorMeasurementProblem<Isometry3> sensor_problem(
        [](int kNumPoses) -> std::vector<Isometry3F64> {
          std::vector<Isometry3F64> path;
          for (int i = 0; i < kNumPoses; ++i) {
            path.push_back(sophus::Isometry3F64::fromTx(0.1 * i));
          }
          return path;
        });
    sensor_problem.test([](Isometry3F64& est) {}, is_near);
    sensor_problem.test(perturb, is_near);
  }

  {
    auto perturb = [](Similarity3F64& est) {
      est.translation() += Eigen::Vector3d(0.3, -0.1, 0.1);
      est.setRotation(est.rotation() * sophus::Rotation3F64::fromRx(0.1));
      est.setScale(est.scale() * 2.0);
    };
    auto is_near = [](Similarity3F64 const& truth,
                      Similarity3F64 const& est) -> Expected<Success> {
      double translation_error =
          (truth.translation() - est.translation()).norm();
      double rotation_error =
          (truth.rotation() * est.rotation().inverse()).log().norm();
      double scale_error = std::abs(truth.scale() - est.scale());
      if (translation_error > kPosThr || rotation_error > kRotThr ||
          scale_error > kScaleThr) {
        return FARM_UNEXPECTED(
            "translation error {} [{}], rotation error {} [{}], scale error "
            "{} "
            "[{}]",
            translation_error,
            kPosThr,
            rotation_error,
            kRotThr,
            scale_error,
            kScaleThr);
      }
      return Success{};
    };
    SimplePriorProblem<Similarity3> prior_problem(Similarity3F64{});
    prior_problem.test([](Similarity3F64& est) {}, is_near);
    prior_problem.test(perturb, is_near);
    TransformGraphProblem<Similarity3, 3> graph_problem(
        [](double p, Similarity3F64& est) {});
    graph_problem.test([](Similarity3F64& est) {}, is_near);
    graph_problem.test(perturb, is_near);
  }

  {
    auto perturb = [](Translation3F64& est) {
      est.translation() += Eigen::Vector3d(0.3, -0.1, 0.1);
    };
    auto is_near = [](Translation3F64 const& truth,
                      Translation3F64 const& est) -> Expected<Success> {
      double translation_error =
          (truth.translation() - est.translation()).norm();
      if (translation_error > kPosThr) {
        return FARM_UNEXPECTED(
            "translation error {} [{}]", translation_error, kPosThr);
      }
      return Success{};
    };
    SimplePriorProblem<Translation3> prior_problem(Translation3F64{});
    prior_problem.test([](Translation3F64& est) {}, is_near);
    prior_problem.test(perturb, is_near);
    TransformGraphProblem<Translation3, 3> graph_problem(
        [](double p, Translation3F64& est) {});
    graph_problem.test([](Translation3F64& est) {}, is_near);
    graph_problem.test(perturb, is_near);
  }
  {
    auto perturb = [](ScalingTranslation3F64& est) {
      est.translation() += Eigen::Vector3d(0.3, -0.1, 0.1);
      est.setScaleFactors(est.scaleFactors() * 2.0);
    };
    auto is_near = [](ScalingTranslation3F64 const& truth,
                      ScalingTranslation3F64 const& est) -> Expected<Success> {
      double translation_error =
          (truth.translation() - est.translation()).norm();
      double scale_error = (truth.scaleFactors() - est.scaleFactors()).norm();
      if (translation_error > kPosThr || scale_error > kScaleThr) {
        return FARM_UNEXPECTED(
            "translation error {} [{}], scale error {} [{}]",
            translation_error,
            kPosThr,
            scale_error,
            kScaleThr);
      }
      return Success{};
    };
    SimplePriorProblem<ScalingTranslation3> prior_problem(
        ScalingTranslation3F64{});
    prior_problem.test([](ScalingTranslation3F64& est) {}, is_near);
    prior_problem.test(perturb, is_near);
    TransformGraphProblem<ScalingTranslation3, 3> graph_problem(
        [](double p, ScalingTranslation3F64& est) {});
    graph_problem.test([](ScalingTranslation3F64& est) {}, is_near);
    graph_problem.test(perturb, is_near);
  }
}

TEST(complex, compile_test) {
  Eigen::Vector2d z0_f64 = Eigen::Vector2d(0.0, 0.0);
  Eigen::Vector2d z1_f64 = Eigen::Vector2d(1.0, 0.0);

  Eigen::Vector2<::ceres::Jet<double, 2>> z_jet(z0_f64[0], z0_f64[1]);

  {
    // double  x  double  =>  double

    Eigen::Vector2d z_f64_r =
        ComplexImpl<double>::multiplication(z0_f64, z1_f64) +
        ComplexImpl<double>::addition(z0_f64, z1_f64);
  }
  {
    // double  x  jet  =>  jet
    Eigen::Vector2<::ceres::Jet<double, 2>> z_jet_r =
        ComplexImpl<double>::multiplication(z1_f64, z_jet) +
        ComplexImpl<double>::addition(z1_f64, z_jet);
  }

  {
    // jet  x  double  =>  jet
    Eigen::Vector2<::ceres::Jet<double, 2>> z_jet_r =
        ComplexImpl<::ceres::Jet<double, 2>>::multiplication(z_jet, z1_f64) +
        ComplexImpl<::ceres::Jet<double, 2>>::addition(z_jet, z1_f64);
  }

  {
    // jet  x  jet  =>  jet
    Eigen::Vector2<::ceres::Jet<double, 2>> z_jet_r =
        ComplexImpl<::ceres::Jet<double, 2>>::multiplication(z_jet, z_jet) +
        ComplexImpl<::ceres::Jet<double, 2>>::addition(z_jet, z_jet);
  }

  auto zz0_f64 = Complex<double>::fromParams(z0_f64);
  auto zz1_f64 = Complex<double>::fromParams(z1_f64);
  auto zz_jet = Complex<::ceres::Jet<double, 2>>::fromParams(z_jet);

  {
    // double  x  double  =>  double
    auto zz_f64_r = zz0_f64 * zz1_f64 + zz0_f64 + zz1_f64;
  }
  {
    // double  x  jet  =>  jet
    auto zz_jet_r = zz1_f64 * zz_jet + zz1_f64 + zz_jet;
  }

  {
    // jet  x  double  =>  jet
    auto zz_jet_r = zz_jet * zz1_f64 + zz_jet + zz1_f64;
  }

  {
    // jet  x  jet  =>  jet
    auto zz_jet_r = zz_jet * zz_jet + zz_jet + zz_jet;
  }
}

TEST(quaternion, compile_test) {
  Eigen::Vector4d z0_f64 = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
  Eigen::Vector4d z1_f64 = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);

  Eigen::Vector4<::ceres::Jet<double, 2>> z_jet =
      Eigen::Vector4<::ceres::Jet<double, 2>>::Zero();

  {
    // double  x  double  =>  double

    Eigen::Vector4d z_f64_r =
        QuaternionImpl<double>::multiplication(z0_f64, z1_f64) +
        QuaternionImpl<double>::addition(z0_f64, z1_f64);
  }
  {
    // double  x  jet  =>  jet
    Eigen::Vector4<::ceres::Jet<double, 2>> z_jet_r =
        QuaternionImpl<double>::multiplication(z1_f64, z_jet) +
        QuaternionImpl<double>::addition(z1_f64, z_jet);
  }

  {
    // jet  x  double  =>  jet
    Eigen::Vector4<::ceres::Jet<double, 2>> z_jet_r =
        QuaternionImpl<::ceres::Jet<double, 2>>::multiplication(z_jet, z1_f64) +
        QuaternionImpl<::ceres::Jet<double, 2>>::addition(z_jet, z1_f64);
  }

  {
    // jet  x  jet  =>  jet
    Eigen::Vector4<::ceres::Jet<double, 2>> z_jet_r =
        QuaternionImpl<::ceres::Jet<double, 2>>::multiplication(z_jet, z_jet) +
        QuaternionImpl<::ceres::Jet<double, 2>>::addition(z_jet, z_jet);
  }

  auto zz0_f64 = Quaternion<double>::fromParams(z0_f64);
  auto zz1_f64 = Quaternion<double>::fromParams(z1_f64);

  auto zz_jet = Quaternion<::ceres::Jet<double, 2>>::fromParams(z_jet);

  {
    // double  x  double  =>  double
    auto zz_f64_r = zz0_f64 * zz1_f64 + zz0_f64 + zz1_f64;
  }
  {
    // double  x  jet  =>  jet
    auto zz_jet_r = zz1_f64 * zz_jet + zz1_f64 + zz_jet;
  }

  {
    // jet  x  double  =>  jet
    auto zz_jet_r = zz_jet * zz1_f64 + zz_jet + zz1_f64;
  }

  {
    // jet  x  jet  =>  jet
    auto zz_jet_r = zz_jet * zz_jet + zz_jet + zz_jet;
  }
}

template <template <class> class TLieGroup>
struct JetLieGroupTests {
  using Jet = ::ceres::Jet<double, 2>;

  template <class TScalar>
  using LieGroup = TLieGroup<TScalar>;
  using LieGroupF64 = LieGroup<double>;
  using LieGroupJet = LieGroup<Jet>;
  using ImplF64 = typename LieGroupF64::Impl;
  using ImplJet = typename LieGroupJet::Impl;

  static int constexpr kNumParams = LieGroupF64::kNumParams;
  static int constexpr kPointDim = LieGroupF64::kPointDim;

  using PointF64 = Eigen::Vector<double, kPointDim>;
  using ParamsF64 = Eigen::Vector<double, kNumParams>;
  using UnitVectorF64 = UnitVector<double, kPointDim>;

  using PointJet = Eigen::Vector<Jet, kPointDim>;
  using ParamsJet = Eigen::Vector<Jet, kNumParams>;
  using UnitVectorJet = UnitVector<Jet, kPointDim>;

  static void testAll() {
    {
      // double  x  double  =>  double
      ParamsF64 r = ImplF64::multiplication(
          ParamsF64::Zero().eval(), ParamsF64::Zero().eval());
    }
    {
      // double  x  jet  =>  jet
      ParamsJet r = ImplF64::multiplication(
          ParamsF64::Zero().eval(), ParamsJet::Zero().eval());
    }
    {
      // jet  x  double  =>  jet
      ParamsJet r = ImplJet::multiplication(
          ParamsJet::Zero().eval(), ParamsF64::Zero().eval());
    }
    {
      // jet  x  jet  =>  jet
      ParamsJet r = ImplJet::multiplication(
          ParamsJet::Zero().eval(), ParamsJet::Zero().eval());
    }

    {
      // double  x  double  =>  double
      PointF64 r =
          ImplF64::action(ParamsF64::Zero().eval(), PointF64::Zero().eval());
    }
    {
      // double  x  jet  =>  jet
      PointJet r =
          ImplF64::action(ParamsF64::Zero().eval(), PointJet::Zero().eval());
    }
    {
      // jet  x  double  =>  jet
      PointJet r =
          ImplJet::action(ParamsJet::Zero().eval(), PointF64::Zero().eval());
    }
    {
      // jet  x  jet  =>  jet
      PointJet r =
          ImplJet::action(ParamsJet::Zero().eval(), PointJet::Zero().eval());
    }

    {
      // double  x  double  =>  double
      UnitVectorF64 r = ImplF64::action(
          ParamsF64::Zero().eval(),
          UnitVectorF64::fromVectorAndNormalize(PointF64::UnitX()));
    }
    {
      // double  x  jet  =>  jet
      UnitVectorJet r = ImplF64::action(
          ParamsF64::Zero().eval(),
          UnitVectorJet::fromVectorAndNormalize(PointJet::UnitX()));
    }
    {
      // jet  x  double  =>  jet
      UnitVectorJet r = ImplJet::action(
          ParamsJet::Zero().eval(),
          UnitVectorF64::fromVectorAndNormalize(PointF64::UnitX()));
    }
    {
      // jet  x  jet  =>  jet
      UnitVectorJet r = ImplJet::action(
          ParamsJet::Zero().eval(),
          UnitVectorJet::fromVectorAndNormalize(PointJet::UnitX()));
    }

    LieGroupF64 g_d = LieGroupF64::elementExamples()[0];
    LieGroupJet g_jet = LieGroupJet::elementExamples()[0];

    {
      // double  x  double  =>  double
      LieGroupF64 g_d_r = g_d * g_d;
    }

    // double  x  jet  =>  jet
    { LieGroupJet g_jet_r = g_d * g_jet; }

    // jet  x  double  =>  jet
    { LieGroupJet g_jet_r = g_jet * g_d; }

    // jet  x  jet  =>  jet
    { LieGroupJet g_jet_r = g_jet * g_jet; }

    PointF64 p_d = PointF64::Random();
    PointJet p_jet = PointJet::Random();

    {
      // double  x  double  =>  double
      PointF64 p_d_r = g_d * p_d;
    }

    // double  x  jet  =>  jet
    { PointJet p_jet_r = g_d * p_jet; }

    // jet  x  double  =>  jet
    { PointJet p_jet_r = g_jet * p_d; }

    // jet  x  jet  =>  jet
    { PointJet p_jet_r = g_jet * p_jet; }

    UnitVectorF64 u_d =
        UnitVectorF64::fromVectorAndNormalize(PointF64::UnitX());
    UnitVectorJet u_jet =
        UnitVectorJet::fromVectorAndNormalize(PointJet::UnitX());

    {
      // double  x  double  =>  double
      UnitVectorF64 u_d_r = g_d * u_d;
    }

    // double  x  jet  =>  jet
    { UnitVectorJet u_jet_r = g_d * u_jet; }

    // jet  x  double  =>  jet
    { UnitVectorJet u_jet_r = g_jet * u_d; }

    // jet  x  jet  =>  jet
    { UnitVectorJet u_jet_r = g_jet * u_jet; }
  }
};

TEST(jet_lie_group, compile_test) {
  JetLieGroupTests<sophus::Identity2>::testAll();
  JetLieGroupTests<sophus::Identity3>::testAll();
  JetLieGroupTests<sophus::Translation2>::testAll();
  JetLieGroupTests<sophus::Translation3>::testAll();

  JetLieGroupTests<sophus::Rotation2>::testAll();
  JetLieGroupTests<sophus::Rotation3>::testAll();
  JetLieGroupTests<sophus::Isometry2>::testAll();
  JetLieGroupTests<sophus::Isometry3>::testAll();

  JetLieGroupTests<sophus::Scaling2>::testAll();
  JetLieGroupTests<sophus::Scaling3>::testAll();
  JetLieGroupTests<sophus::ScalingTranslation2>::testAll();
  JetLieGroupTests<sophus::ScalingTranslation3>::testAll();

  JetLieGroupTests<sophus::SpiralSimilarity2>::testAll();
  JetLieGroupTests<sophus::SpiralSimilarity3>::testAll();
  JetLieGroupTests<sophus::Similarity2>::testAll();
  JetLieGroupTests<sophus::Similarity3>::testAll();
}

}  // namespace sophus::test
