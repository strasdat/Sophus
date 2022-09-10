// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/rxso2.h"

#include "sophus/ceres/details/test_impl.h"

#include <ceres/ceres.h>

#include <iostream>

template <typename TT>
using StdVector = std::vector<TT, Eigen::aligned_allocator<TT>>;

template <>
struct RotationalPart<sophus::RxSO2d> {
  static double norm(const typename sophus::RxSO2d::Tangent &t) {
    return std::abs(t[0]);
  }
};

int main(int /*unused*/, char ** /*unused*/) {
  using RxSO2d = sophus::RxSO2d;
  using Tangent = RxSO2d::Tangent;
  using Point = RxSO2d::Point;
  double const k_pi = sophus::kPi<double>;

  StdVector<RxSO2d> rxso2_vec;
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0.2, 1.)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0.2, 1.1)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0., 1.1)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0.00001, 0.)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0.00001, 0.00001)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(k_pi, 0.9)));
  rxso2_vec.push_back(
      RxSO2d::exp(Tangent(0.2, 0)) * RxSO2d::exp(Tangent(k_pi, 0.0)) *
      RxSO2d::exp(Tangent(-0.2, 0)));
  rxso2_vec.push_back(
      RxSO2d::exp(Tangent(0.3, 0)) * RxSO2d::exp(Tangent(k_pi, 0.001)) *
      RxSO2d::exp(Tangent(-0.3, 0)));

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

  std::cerr << "Test Ceres RxSo2" << std::endl;
  sophus::LieGroupCeresTests<sophus::RxSo2>(rxso2_vec, point_vec).testAll();
  return 0;
}
