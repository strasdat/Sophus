// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/so2.h"

#include "sophus/ceres/details/test_impl.h"

#include <ceres/ceres.h>

#include <iostream>

template <class TScalar>
using StdVector = std::vector<TScalar, Eigen::aligned_allocator<TScalar>>;

template <>
struct RotationalPart<sophus::SO2d> {
  static double norm(const typename sophus::SO2d::Tangent &t) {
    return std::abs(t);
  }
};

int main(int /*unused*/, char ** /*unused*/) {
  using SO2d = sophus::SO2d;
  using Point = SO2d::Point;
  double const k_pi = sophus::kPi<double>;

  StdVector<SO2d> so2_vec;
  so2_vec.emplace_back(SO2d::exp(0.0));
  so2_vec.emplace_back(SO2d::exp(0.2));
  so2_vec.emplace_back(SO2d::exp(10.));
  so2_vec.emplace_back(SO2d::exp(0.00001));
  so2_vec.emplace_back(SO2d::exp(k_pi));
  so2_vec.emplace_back(SO2d::exp(0.2) * SO2d::exp(k_pi) * SO2d::exp(-0.2));
  so2_vec.emplace_back(SO2d::exp(-0.3) * SO2d::exp(k_pi) * SO2d::exp(0.3));

  StdVector<Point> point_vec;
  point_vec.emplace_back(Point(1.012, 2.73));
  point_vec.emplace_back(Point(9.2, -7.3));
  point_vec.emplace_back(Point(2.5, 0.1));
  point_vec.emplace_back(Point(12.3, 1.9));
  point_vec.emplace_back(Point(-3.21, 3.42));
  point_vec.emplace_back(Point(-8.0, 6.1));
  point_vec.emplace_back(Point(0.0, 2.5));
  point_vec.emplace_back(Point(7.1, 7.8));
  point_vec.emplace_back(Point(5.8, 9.2));

  std::cerr << "Test Ceres So2" << std::endl;
  sophus::LieGroupCeresTests<sophus::So2>(so2_vec, point_vec).testAll();
  return 0;
}
