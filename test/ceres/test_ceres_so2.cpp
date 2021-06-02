#include <ceres/ceres.h>
#include <iostream>
#include <sophus/so2.hpp>

#include "tests.hpp"

template <typename T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

int main(int, char**) {
  using SO2d = Sophus::SO2d;
  using Point = SO2d::Point;
  double const kPi = Sophus::Constants<double>::pi();

  StdVector<SO2d> so2_vec = {SO2d::exp(0.0),
                             SO2d::exp(0.2),
                             SO2d::exp(10.),
                             SO2d::exp(0.00001),
                             SO2d::exp(kPi),
                             SO2d::exp(0.2) * SO2d::exp(kPi) * SO2d::exp(-0.2),
                             SO2d::exp(-0.3) * SO2d::exp(kPi) * SO2d::exp(0.3)};

  StdVector<Point> point_vec = {
      Point(1.012, 2.73), Point(9.2, -7.3),   Point(2.5, 0.1),
      Point(12.3, 1.9),   Point(-3.21, 3.42), Point(-8.0, 6.1),
      Point(0.0, 2.5),    Point(7.1, 7.8),    Point(5.8, 9.2)};

  std::cerr << "Test Ceres SO2" << std::endl;
  Sophus::LieGroupCeresTests<Sophus::SO2>(so2_vec, point_vec).testAll();
  return 0;
}
