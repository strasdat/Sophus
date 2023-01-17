// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/sim3.h"

#include "sophus/ceres/details/test_impl.h"

#include <ceres/ceres.h>

#include <iostream>

template <class TT>
using StdVector = std::vector<TT, Eigen::aligned_allocator<TT>>;

template <>
struct RotationalPart<sophus::Sim3d> {
  static double norm(const typename sophus::Sim3d::Tangent &t) {
    return t.template segment<3>(3).norm();
  }
};

int main(int /*unused*/, char ** /*unused*/) {
  using RxSO3d = sophus::RxSO3d;
  using Sim3d = sophus::Sim3d;
  using Point = Sim3d::Point;
  using Vector4d = Eigen::Vector4d;
  double const k_pi = sophus::kPi<double>;

  StdVector<Sim3d> sim3_vec;
  sim3_vec.push_back(
      Sim3d(RxSO3d::exp(Vector4d(0.2, 0.5, 0.0, 1.)), Point(0, 0, 0)));
  sim3_vec.push_back(
      Sim3d(RxSO3d::exp(Vector4d(0.2, 0.5, -1.0, 1.1)), Point(0.1, 0, 0)));
  sim3_vec.push_back(
      Sim3d(RxSO3d::exp(Vector4d(0., 0., 0., 0.001)), Point(0, 0.1, 0.5)));
  sim3_vec.push_back(
      Sim3d(RxSO3d::exp(Vector4d(0., 0., 0., 1.1)), Point(0, 0.1, 0.5)));
  sim3_vec.push_back(
      Sim3d(RxSO3d::exp(Vector4d(0., 0., 0.00001, 0.)), Point(0, 0, 0)));
  sim3_vec.push_back(Sim3d(
      RxSO3d::exp(Vector4d(0., 0., 0.00001, 0.0000001)),
      Point(1, -1.00000001, 0.2000000001)));
  sim3_vec.push_back(
      Sim3d(RxSO3d::exp(Vector4d(0., 0., 0.00001, 0)), Point(0.01, 0, 0)));
  sim3_vec.push_back(
      Sim3d(RxSO3d::exp(Vector4d(k_pi, 0, 0, 0.9)), Point(0.4, -0.5, 0)));
  sim3_vec.push_back(
      Sim3d(RxSO3d::exp(Vector4d(0.2, 0.5, 0.0, 0)), Point(0, 0, 0)) *
      Sim3d(RxSO3d::exp(Vector4d(k_pi, 0, 0, 0)), Point(0, 0, 0)) *
      Sim3d(RxSO3d::exp(Vector4d(-0.2, -0.5, -0.0, 0)), Point(0, 0, 0)));
  sim3_vec.push_back(
      Sim3d(RxSO3d::exp(Vector4d(0.3, 0.5, 0.1, 0)), Point(0.2, 0, -0.7)) *
      Sim3d(RxSO3d::exp(Vector4d(k_pi, 0, 0, 0)), Point(0, 0, 0)) *
      Sim3d(RxSO3d::exp(Vector4d(-0.3, -0.5, -0.1, 0)), Point(0, 0.6, 0)));

  StdVector<Point> point_vec;
  point_vec.push_back(Point(1.012, 2.73, -1.4));
  point_vec.push_back(Point(0.92, -0.73, -4.4));
  point_vec.push_back(Point(2.5, 0.1, 0.91));
  point_vec.push_back(Point(0.123, 1.9, 0.38));
  point_vec.push_back(Point(-3.21, 3.42, 2.3));
  point_vec.push_back(Point(-8.0, 6.1, -1.1));
  point_vec.push_back(Point(0.0, 2.5, 0.59));
  point_vec.push_back(Point(0.71, 0.78, -0.14));
  point_vec.push_back(Point(0.58, 0.92, 0.0));

  std::cerr << "Test Ceres Sim3" << std::endl;
  sophus::LieGroupCeresTests<sophus::Sim3>(sim3_vec, point_vec).testAll();
  return 0;
}
