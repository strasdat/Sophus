// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/se3.h"

#include "sophus/ceres/details/test_impl.h"

#include <ceres/ceres.h>

#include <iostream>

template <class TT>
using StdVector = std::vector<TT, Eigen::aligned_allocator<TT>>;

template <>
struct RotationalPart<sophus::SE3d> {
  static double norm(const typename sophus::SE3d::Tangent &t) {
    return t.template tail<3>().norm();
  }
};

int main(int /*unused*/, char ** /*unused*/) {
  using SE3d = sophus::SE3d;
  using So3F64 = sophus::So3F64;
  using Point = SE3d::Point;
  double const k_pi = sophus::kPi<double>;

  StdVector<SE3d> se3_vec;
  se3_vec.push_back(SE3d(So3F64::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)));
  se3_vec.push_back(SE3d(So3F64::exp(Point(0.2, 0.5, -1.0)), Point(10, 0, 0)));
  se3_vec.push_back(SE3d(So3F64::exp(Point(0., 0., 0.)), Point(0, 100, 5)));
  se3_vec.push_back(SE3d(So3F64::exp(Point(0., 0., 0.00001)), Point(0, 0, 0)));
  se3_vec.push_back(SE3d(
      So3F64::exp(Point(0., 0., 0.00001)),
      Point(0, -0.00000001, 0.0000000001)));
  se3_vec.push_back(
      SE3d(So3F64::exp(Point(0., 0., 0.00001)), Point(0.01, 0, 0)));
  se3_vec.push_back(SE3d(So3F64::exp(Point(k_pi, 0, 0)), Point(4, -5, 0)));
  se3_vec.push_back(
      SE3d(So3F64::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)) *
      SE3d(So3F64::exp(Point(k_pi, 0, 0)), Point(0, 0, 0)) *
      SE3d(So3F64::exp(Point(-0.2, -0.5, -0.0)), Point(0, 0, 0)));
  se3_vec.push_back(
      SE3d(So3F64::exp(Point(0.3, 0.5, 0.1)), Point(2, 0, -7)) *
      SE3d(So3F64::exp(Point(k_pi, 0, 0)), Point(0, 0, 0)) *
      SE3d(So3F64::exp(Point(-0.3, -0.5, -0.1)), Point(0, 6, 0)));

  StdVector<Point> point_vec;
  point_vec.push_back(Point(1.012, 2.73, -1.4));
  point_vec.push_back(Point(9.2, -7.3, -4.4));
  point_vec.push_back(Point(2.5, 0.1, 9.1));
  point_vec.push_back(Point(12.3, 1.9, 3.8));
  point_vec.push_back(Point(-3.21, 3.42, 2.3));
  point_vec.push_back(Point(-8.0, 6.1, -1.1));
  point_vec.push_back(Point(0.0, 2.5, 5.9));
  point_vec.push_back(Point(7.1, 7.8, -14));
  point_vec.push_back(Point(5.8, 9.2, 0.0));

  std::cerr << "Test Ceres Se3" << std::endl;
  sophus::LieGroupCeresTests<sophus::Se3>(se3_vec, point_vec).testAll();
  return 0;
}
