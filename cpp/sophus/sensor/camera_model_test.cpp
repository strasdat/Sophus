// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/sensor/camera_model.h"

#include "sophus/calculus/num_diff.h"
#include "sophus/image/interpolation.h"
#include "sophus/lie/se3.h"
#include "sophus/sensor/orthographic.h"

#include <farm_ng/core/logging/eigen.h>
#include <farm_ng/core/logging/logger.h>
#include <gtest/gtest.h>

using namespace sophus;

double constexpr kEps = 1e-5;

CameraModel openCvCameraModel() {
  int w = 848;
  int h = 800;
  double fx = 286;
  double fy = fx;
  double cx = (w - 1.0) * 0.5;
  double cy = (h - 1.0) * 0.5;
  Eigen::Matrix<double, 9, 1> get_params;
  get_params << fx, fy, cx, cy, -0.0080617731437087059, 0.04318523034453392,
      -0.039864420890808105, 0.0068964879028499126, 0.0;
  return CameraModel(BrownConradyModel({w, h}, get_params));
}

// TODO: move to farm_ng_utils

// Eigen::Vector2d findImagePoint(
//     cv::Mat image, int near_x, int near_y, int window) {
//   FARM_CHECK_EQ(image.type(), CV_32FC1);
//   double x_res = 0;
//   double y_res = 0;
//   double total = 0;
//   for (int i = -window; i <= window; ++i) {
//     for (int j = -window; j <= window; ++j) {
//       double pixel =
//           static_cast<double>(image.at<float>(i + near_y, j + near_x));
//       if (pixel < 1) {
//         y_res += static_cast<double>(i + near_y) * (1 - pixel);
//         x_res += static_cast<double>(j + near_x) * (1 - pixel);
//         total += (1 - pixel);
//       }
//     }
//   }
//   if (total == 0) {
//     return Eigen::Vector2d(-1, -1);
//   }
//   return Eigen::Vector2d(x_res / total, y_res / total);
// }

TEST(camera_model, projection_round_trip) {
  std::vector<CameraModel> camera_models;
  CameraModel pinhole = CameraModel::createDefaultPinholeModel({640, 480});
  Eigen::VectorXd get_params(8);
  get_params << 1000, 1000, 320, 280, 0.1, 0.01, 0.001, 0.0001;
  CameraModel kb3 = CameraModel(
      {640, 480}, CameraDistortionType::kannala_brandt_k3, get_params);

  camera_models.push_back(pinhole);
  camera_models.push_back(kb3);
  camera_models.push_back(openCvCameraModel());

  for (CameraModel const& camera_model : camera_models) {
    std::vector<Eigen::Vector2d> pixels_image = {
        {0, 0}, {1, 400}, {320, 240}, {319.5, 239.5}, {100, 40}, {639, 479}};

    Image<Eigen::Vector2f> unwarp_table = camera_model.undistortTable();

    for (auto const& pixel_image : pixels_image) {
      for (double d : {0.1, 0.5, 1.0, 1.1, 3.0, 15.0}) {
        Eigen::Vector3d point_in_camera =
            camera_model.camUnproj(pixel_image, d);
        FARM_CHECK_NEAR(d, point_in_camera.z(), kEps);

        Eigen::Vector2d pixel_image2 = camera_model.camProj(point_in_camera);

        EXPECT_NEAR(pixel_image.x(), pixel_image2.x(), kEps);
        EXPECT_NEAR(pixel_image.y(), pixel_image2.y(), kEps);

        Eigen::Matrix<double, 2, 6> dx =
            camera_model.dxCamProjExpXPointAt0(point_in_camera);
        Eigen::Vector<double, 6> zero;
        zero.setZero();
        Eigen::Matrix<double, 2, 6> numeric_dx =
            sophus::vectorFieldNumDiff<double, 2, 6>(
                [&](Eigen::Vector<double, 6> const& x) -> Eigen::Vector2d {
                  return camera_model.camProj(
                      sophus::SE3d::exp(x) * point_in_camera);
                },
                zero);
        FARM_CHECK_NEAR(dx, numeric_dx, 1e-2);
      }

      Eigen::Vector2d ab_in_z1plane = camera_model.undistort(pixel_image);
      Eigen::Vector2d ab_in_z1plane2 =
          interpolate(unwarp_table, pixel_image.cast<float>()).cast<double>();
      FARM_CHECK_NEAR(ab_in_z1plane, ab_in_z1plane2, kEps);

      Eigen::Vector2d pixel_in_image = camera_model.distort(ab_in_z1plane);

      FARM_CHECK_NEAR(pixel_image, pixel_in_image, kEps);

      Eigen::Matrix2d dx = camera_model.dxDistort(ab_in_z1plane);

      Eigen::Matrix2d dx_num = sophus::vectorFieldNumDiff<double, 2, 2>(
          [&](Eigen::Vector2d const& x) -> Eigen::Vector2d {
            return camera_model.distort(x);
          },
          ab_in_z1plane);

      // FARM_LOG_INFO(
      //     "name: {} point: \n{}\ndx: \n{}",
      //     camera_model.frameName(),
      //     ab_in_z1plane,
      //     dx);

      FARM_CHECK_NEAR(dx, dx_num, kEps);
    }
  }

  std::vector<Eigen::Vector<double, 6>> tangent_vec;

  Eigen::Vector<double, 6> tmp;
  tmp << 0, 0, 0, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 0.1, 0, 0, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 0, 0.1, 0, 0.1, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << 0, -0.5, 0.1, 0, 0, 0;
  tangent_vec.push_back(tmp);
  tmp << -0.1, 0.1, 2, 0, 0, 0.1;
  tangent_vec.push_back(tmp);
  tmp << 0.2, -0.1, 1, -0.1, 0.1, 0;
  tangent_vec.push_back(tmp);
  tmp << 0.3, 0.5, 0.1, 0.2, -0.1, 0;
  tangent_vec.push_back(tmp);

  for (CameraModel const& camera_model : camera_models) {
    for (Eigen::Vector<double, 6> const& t : tangent_vec) {
      sophus::SE3d foo_pose_bar = sophus::SE3d::exp(t);

      PointTransformer trans(foo_pose_bar);
      std::vector<Eigen::Vector2d> pixels_image = {
          {0, 0}, {1, 400}, {320, 240}, {319.5, 239.5}, {100, 40}, {639, 479}};

      for (auto const& pixel_image : pixels_image) {
        for (double d : {0.1, 0.5, 1.0, 1.1, 3.0, 15.0}) {
          Eigen::Vector3d point_in_bar_camera =
              camera_model.camUnproj(pixel_image, d);

          InverseDepthPoint3F64 inverse_depth_in_bar =
              InverseDepthPoint3F64::fromEuclideanPoint3(point_in_bar_camera);

          Eigen::Matrix<double, 2, 6> dx =
              camera_model.dxDistort(proj(foo_pose_bar * point_in_bar_camera)) *
              trans.dxProjExpXTransformPointAt0(inverse_depth_in_bar);

          Eigen::Vector<double, 6> zero;
          zero.setZero();
          Eigen::Matrix<double, 2, 6> const num_dx =
              vectorFieldNumDiff<double, 2, 6>(
                  [&](Eigen::Vector<double, 6> const& vec_a) {
                    return camera_model.distort(proj(
                        sophus::Se3F64::exp(vec_a) * foo_pose_bar *
                        point_in_bar_camera));
                  },
                  zero);

          FARM_CHECK_NEAR(dx, num_dx, 0.005);

          {
            Eigen::Matrix<double, 2, 3> dx =
                camera_model.dxDistort(
                    proj(foo_pose_bar * point_in_bar_camera)) *
                trans.dxProjTransformX(inverse_depth_in_bar);

            Eigen::Matrix<double, 2, 3> const num_dx =
                vectorFieldNumDiff<double, 2, 3>(
                    [&](Eigen::Vector<double, 3> const& ab_psi) {
                      return camera_model.camProj(trans.scaledTransform(
                          InverseDepthPoint3F64::fromAbAndPsi(ab_psi)));
                    },
                    inverse_depth_in_bar.params());
            FARM_CHECK_NEAR(dx, num_dx, kEpsilonSqrtF64);
          }
        }
      }
    }
  }
}

// TODO: move to farm_ng_utils / or opencv deps just for the tests

// TEST(camera_model, brown_conrady_compare_to_opencv) {
//   CameraModel opencv_model = openCvCameraModel();

//   cv::Mat dist_coeffs(5, 1, cv::DataType<double>::type);
//   for (int i = 0; i < 5; ++i) {
//     dist_coeffs.at<double>(i) = opencv_model.distortionParams()[i];
//   }

//   std::vector<Eigen::Vector2d> pixels_image = {
//       {0, 0}, {1, 400}, {100, 40}, {639, 479}};

//   cv::Mat intrinsics_mat(3, 3, cv::DataType<double>::type);
//   intrinsics_mat.at<double>(0, 0) = opencv_model.params()[0];
//   intrinsics_mat.at<double>(1, 0) = 0.0;
//   intrinsics_mat.at<double>(2, 0) = 0.0;

//   intrinsics_mat.at<double>(0, 1) = 0.0;
//   intrinsics_mat.at<double>(1, 1) = opencv_model.params()[1];
//   intrinsics_mat.at<double>(2, 1) = 0.0;

//   intrinsics_mat.at<double>(0, 2) = opencv_model.params()[2];
//   intrinsics_mat.at<double>(1, 2) = opencv_model.params()[3];
//   intrinsics_mat.at<double>(2, 2) = 1.0;
//   for (const auto& pixel_image : pixels_image) {
//     Eigen::Vector3d point_image = opencv_model.camUnproj(pixel_image, 1.0);
//     Eigen::Vector2d pixel_image2 = opencv_model.camProj(point_image);

//     std::vector<cv::Point3d> cv_point_image = {
//         cv::Point3d(point_image.x(), point_image.y(), point_image.z())};

//     std::vector<cv::Point2d> cv_pixel_out;

//     cv::Mat vec(3, 1, cv::DataType<double>::type);
//     vec.at<double>(0) = 0.0;
//     vec.at<double>(1) = 0.0;
//     vec.at<double>(2) = 0.0;

//     cv::projectPoints(
//         cv_point_image, vec, vec, intrinsics_mat, dist_coeffs, cv_pixel_out);
//     EXPECT_NEAR(cv_pixel_out[0].x, pixel_image2.x(), kEps);
//     EXPECT_NEAR(cv_pixel_out[0].y, pixel_image2.y(), kEps);
//   }
// }

// TEST(camera_model, scale_up_down) {
//   {
//     // Proof: pixel 4 on level 0 maps to pixel 2 on level 1.
//     cv::Mat img(cv::Size(12, 1), CV_32FC1, 0.0);
//     img.at<float>(0, 4) = 1.f;
//     std::vector<cv::Mat> img_pyr;
//     cv::buildPyramid(img, img_pyr, 2);

//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 0), 0.0);
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 1), 0.0625);
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 2), 0.375);  // pixel 2 at level 1
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 3), 0.0625);
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 4), 0.0);
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 5), 0.0);
//   }
//   {
//     // Proof: pixel 5 on level 0 maps to pixel 2.5 on level 1.
//     cv::Mat img(cv::Size(12, 1), CV_32FC1, 0.0);

//     img.at<float>(0, 5) = 1.f;
//     std::vector<cv::Mat> img_pyr;
//     cv::buildPyramid(img, img_pyr, 2);

//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 0), 0.0);
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 1), 0.0);
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 2), 0.25);  // pixel 2 at level 1
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 3), 0.25);  // pixel 3 at level 1
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 4), 0.0);
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 5), 0.0);
//   }
//   {
//     // Proof: pixel 6 on level 0 maps to pixel 3 on level 1.
//     cv::Mat img(cv::Size(12, 1), CV_32FC1, 0.0);

//     img.at<float>(0, 6) = 1.f;
//     std::vector<cv::Mat> img_pyr;
//     cv::buildPyramid(img, img_pyr, 2);

//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 0), 0.0);
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 1), 0.0);
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 2), 0.0625);
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 3), 0.375);  // pixel 3 at level 1
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 4), 0.0625);
//     FARM_CHECK_EQ(img_pyr[1].at<float>(0, 5), 0.0);
//   }

//   std::vector<CameraModel> camera_models;
//   CameraModel pinhole =
//       CameraModel::createDefaultPinholeModel("pinhole", {640, 480});
//   Eigen::VectorXd get_params(8);
//   get_params << 1000, 1000, 320, 280, 0.1, 0.01, 0.001, 0.0001;
//   CameraModel kb3 = CameraModel(
//       "pinhole",
//       {640, 480},
//       CameraDistortionType::kannala_brandt_k3,
//       get_params);

//   camera_models.push_back(pinhole);
//   camera_models.push_back(kb3);
//   camera_models.push_back(openCvCameraModel());

//   for (const CameraModel& camera_model : camera_models) {
//     std::vector<Eigen::Vector2d> pixels_lvl0 = {{0, 0}, {4, 0}, {8, 4}};
//     std::vector<Eigen::Vector2d> pixels_lvl1 = {{0, 0}, {2, 0}, {4, 2}};

//     FARM_CHECK_EQ(pixels_lvl0.size(), pixels_lvl1.size());

//     for (size_t i = 0; i < pixels_lvl0.size(); ++i) {
//       Eigen::Vector3d point_image =
//       camera_model.camUnproj(pixels_lvl0[i], 1.0); Eigen::Vector3d
//       point_image2 =
//           camera_model.subsampleDown().camUnproj(pixels_lvl1[i], 1.0);

//       FARM_CHECK_EQ(
//           camera_model.imageSize().width,
//           camera_model.subsampleDown().imageSize().width * 2);
//       FARM_CHECK_EQ(
//           camera_model.imageSize().height,
//           camera_model.subsampleDown().imageSize().height * 2);
//       for (int r = 0; r < 3; ++r) {
//         EXPECT_NEAR(point_image[r], point_image2[r], 1e-5);
//       }
//     }
//   }
// }

TEST(camera_model, scale_up_down_roundtrip) {
  CameraModel pinhole = CameraModel::createDefaultPinholeModel({640, 480});

  CameraModel pinhole_binned_down = pinhole.binDown();
  FARM_CHECK_EQ(pinhole_binned_down.imageSize(), ImageSize(320, 240));
  CameraModel pinhole_binned_down_and_up = pinhole_binned_down.binUp();
  FARM_CHECK_EQ(pinhole.params(), pinhole_binned_down_and_up.params());
  FARM_CHECK_EQ(pinhole.imageSize(), pinhole_binned_down_and_up.imageSize());

  CameraModel pinhole_subsampled_down = pinhole.subsampleDown();
  FARM_CHECK_EQ(pinhole_subsampled_down.imageSize(), ImageSize(320, 240));
  CameraModel pinhole_subsample_down_and_up =
      pinhole_subsampled_down.subsampleUp();
  FARM_CHECK_EQ(pinhole.params(), pinhole_subsample_down_and_up.params());
  FARM_CHECK_EQ(pinhole.imageSize(), pinhole_subsample_down_and_up.imageSize());
}

TEST(camera_model, ortho_cam) {
  Eigen::AlignedBox2d bounding_box;
  bounding_box.min().x() = -1.5;
  bounding_box.min().y() = -1.0;
  bounding_box.max().x() = 1.5;
  bounding_box.max().y() = 1.0;

  OrthographicModel ortho_cam =
      orthoCamFromBoundingBox(bounding_box, ImageSize(600, 400));
  EXPECT_NEAR(ortho_cam.scaleX(), 0.005, 1e-9);
  EXPECT_NEAR(ortho_cam.scaleY(), 0.005, 1e-9);
  EXPECT_NEAR(ortho_cam.offsetX(), -1.4975, 1e-9);
  EXPECT_NEAR(ortho_cam.offsetY(), -0.9975, 1e-9);

  std::vector<Eigen::Vector2d> obs(
      {Eigen::Vector2d(-0.5, -0.5),
       Eigen::Vector2d(-0.5, 400 - 0.5),
       Eigen::Vector2d(600 - 0.5, -0.5),
       Eigen::Vector2d(600 - 0.5, 400 - 0.5)});
  std::vector<Eigen::Vector2d> coords(
      {Eigen::Vector2d(-1.5, -1.0),
       Eigen::Vector2d(-1.5, 1.0),
       Eigen::Vector2d(1.5, -1.0),
       Eigen::Vector2d(1.5, 1.0)});

  for (size_t i = 0; i < obs.size(); ++i) {
    Eigen::Vector2d uv = obs[i];
    Eigen::Vector2d coord = coords[i];

    Eigen::Vector2d coord2 = ortho_cam.camUnproj(uv, 1.0).head<2>();
    EXPECT_NEAR(coord.x(), coord2.x(), 1e-5);
    EXPECT_NEAR(coord.y(), coord2.y(), 1e-5);
  }

  Eigen::AlignedBox2d bounding_box2 = boundingBoxFromOrthoCam(ortho_cam);

  EXPECT_NEAR(bounding_box.min().x(), bounding_box2.min().x(), 1e-5);
  EXPECT_NEAR(bounding_box.min().y(), bounding_box2.min().y(), 1e-5);
  EXPECT_NEAR(bounding_box.max().x(), bounding_box2.max().x(), 1e-5);
  EXPECT_NEAR(bounding_box.max().y(), bounding_box2.max().y(), 1e-5);
}

// TEST(camera_model, binDown) {
//   // Full resolution source data
//   cv::Mat img_src = cv::Mat(cv::Size(16, 16), CV_32FC1, cv::Scalar(1.0));
//   int x_img_src = 8;
//   int y_img_src = 8;
//   for (int i = -1; i <= 1; ++i) {
//     for (int j = -1; j <= 1; ++j) {
//       img_src.at<float>(cv::Point(i + x_img_src, j + y_img_src)) = 0.0;
//     }
//   }
//   CameraModel pinhole_src =
//       CameraModel::createDefaultPinholeModel("pinhole", {16, 16});
//   double z = 10.0;
//   Eigen::Vector3d xyz =
//       pinhole_src.camUnproj(Eigen::Vector2d(x_img_src, y_img_src), z);

//   // Test subsample path
//   cv::Mat img_dst_subsample;
//   cv::pyrDown(img_src, img_dst_subsample);
//   CameraModel pinhole_subsample_down = pinhole_src.subsampleDown();
//   Eigen::Vector2d img_pt_subsample_proj =
//   pinhole_subsample_down.camProj(xyz); Eigen::Vector2d img_pt_subsample =
//   findImagePoint(img_dst_subsample, 4, 4, 2);
//   EXPECT_NEAR(img_pt_subsample(0), img_pt_subsample_proj(0), 0.1);
//   EXPECT_NEAR(img_pt_subsample(1), img_pt_subsample_proj(1), 0.1);

//   // Test binned path
//   cv::Mat img_dst_binned = binImageDown(img_src);
//   CameraModel pinhole_binned_down = pinhole_src.binDown();
//   Eigen::Vector2d img_pt_bin_proj = pinhole_binned_down.camProj(xyz);
//   Eigen::Vector2d img_pt_bin = findImagePoint(img_dst_binned, 4, 4, 2);
//   EXPECT_NEAR(img_pt_bin(0), img_pt_bin_proj(0), 0.1);
//   EXPECT_NEAR(img_pt_bin(1), img_pt_bin_proj(1), 0.1);
// }
