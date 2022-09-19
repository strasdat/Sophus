// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/se2.h"

#include "sophus/ceres/details/test_impl.h"

#include <ceres/ceres.h>

#include <iostream>

template <typename TT>
using StdVector = std::vector<TT, Eigen::aligned_allocator<TT>>;

template <>
struct RotationalPart<sophus::Se2F64> {
  static double norm(const typename sophus::Se2F64::Tangent &t) {
    return std::abs(t[2]);
  }
};

int main(int /*unused*/, char ** /*unused*/) {
  using Se2F64 = sophus::Se2F64;
  using SO2d = sophus::SO2d;
  using Point = Se2F64::Point;
  double const k_pi = sophus::kPi<double>;

  StdVector<Se2F64> se2_vec;
  se2_vec.push_back(Se2F64(SO2d(0.0), Point(0, 0)));
  se2_vec.push_back(Se2F64(SO2d(0.2), Point(10, 0)));
  se2_vec.push_back(Se2F64(SO2d(0.), Point(0, 100)));
  se2_vec.push_back(Se2F64(SO2d(-1.), Point(20, -1)));
  se2_vec.push_back(Se2F64(SO2d(0.00001), Point(-0.00000001, 0.0000000001)));
  se2_vec.push_back(
      Se2F64(SO2d(0.2), Point(0, 0)) * Se2F64(SO2d(k_pi), Point(0, 0)) *
      Se2F64(SO2d(-0.2), Point(0, 0)));
  se2_vec.push_back(
      Se2F64(SO2d(0.3), Point(2, 0)) * Se2F64(SO2d(k_pi), Point(0, 0)) *
      Se2F64(SO2d(-0.3), Point(0, 6)));

  StdVector<Point> point_vec;
  point_vec.push_back(Point(1.012, 2.73));
  point_vec.push_back(Point(9.2, -7.3));
  point_vec.push_back(Point(2.5, 0.1));
  point_vec.push_back(Point(12.3, 1.9));
  point_vec.push_back(Point(-3.21, 3.42));
  point_vec.push_back(Point(-8.0, 6.1));
  point_vec.push_back(Point(0.0, 2.5));
  point_vec.push_back(Point(7.1, 7.8));
  point_vec.push_back(Point(5.8, 9.2));

  std::cerr << "Test Ceres Se2" << std::endl;
  sophus::LieGroupCeresTests<sophus::Se2>(se2_vec, point_vec).testAll();
  return 0;
}
