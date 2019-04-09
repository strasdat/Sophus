#include <ceres/ceres.h>
#include <iostream>
#include <sophus/rxso3.hpp>

#include "tests.hpp"

template <typename T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

int main(int, char**) {
  using RxSO3d = Sophus::RxSO3d;
  using Point = RxSO3d::Point;
  using Tangent = RxSO3d::Tangent;
  double const kPi = Sophus::Constants<double>::pi();

  StdVector<RxSO3d> rxso3_vec = {RxSO3d::exp(Tangent(0.2, 0.5, 0.0, 1.)),
                                 RxSO3d::exp(Tangent(0.2, 0.5, -1.0, 1.1)),
                                 RxSO3d::exp(Tangent(0., 0., 0., 1.1)),
                                 RxSO3d::exp(Tangent(0., 0., 0.00001, 0.)),
                                 RxSO3d::exp(Tangent(0., 0., 0.00001, 0.00001)),
                                 RxSO3d::exp(Tangent(0., 0., 0.00001, 0)),

                                 RxSO3d::exp(Tangent(kPi, 0, 0, 0.9)),

                                 RxSO3d::exp(Tangent(0.2, -0.5, 0, 0)) *
                                     RxSO3d::exp(Tangent(kPi, 0, 0, 0)) *
                                     RxSO3d::exp(Tangent(-0.2, -0.5, 0, 0)),

                                 RxSO3d::exp(Tangent(0.3, 0.5, 0.1, 0)) *
                                     RxSO3d::exp(Tangent(kPi, 0, 0, 0)) *
                                     RxSO3d::exp(Tangent(-0.3, -0.5, -0.1, 0))};

  StdVector<Point> point_vec = {
      Point(1.012, 2.73, -1.4), Point(9.2, -7.3, -4.4),  Point(2.5, 0.1, 9.1),
      Point(12.3, 1.9, 3.8),    Point(-3.21, 3.42, 2.3), Point(-8.0, 6.1, -1.1),
      Point(0.0, 2.5, 5.9),     Point(7.1, 7.8, -14),    Point(5.8, 9.2, 0.0)};

  Sophus::LieGroupCeresTests<Sophus::RxSO3>(rxso3_vec, point_vec).testAll();
  return 0;
}
