#include <ceres/ceres.h>
#include <iostream>
#include <sophus/se2.hpp>

#include "tests.hpp"

template <typename T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

int main(int, char**) {
  using SE2d = Sophus::SE2d;
  using SO2d = Sophus::SO2d;
  using Point = SE2d::Point;
  double const kPi = Sophus::Constants<double>::pi();

  StdVector<SE2d> se2_vec = {
      SE2d(SO2d(0.0), Point(0, 0)),
      SE2d(SO2d(0.2), Point(10, 0)),
      SE2d(SO2d(0.), Point(0, 100)),
      SE2d(SO2d(-1.), Point(20, -1)),
      SE2d(SO2d(0.00001), Point(-0.00000001, 0.0000000001)),
      SE2d(SO2d(0.2), Point(0, 0)) * SE2d(SO2d(kPi), Point(0, 0)) *
          SE2d(SO2d(-0.2), Point(0, 0)),
      SE2d(SO2d(0.3), Point(2, 0)) * SE2d(SO2d(kPi), Point(0, 0)) *
          SE2d(SO2d(-0.3), Point(0, 6))};

  StdVector<Point> point_vec = {
      Point(1.012, 2.73), Point(9.2, -7.3),   Point(2.5, 0.1),
      Point(12.3, 1.9),   Point(-3.21, 3.42), Point(-8.0, 6.1),
      Point(0.0, 2.5),    Point(7.1, 7.8),    Point(5.8, 9.2)};

  std::cerr << "Test Ceres SE2" << std::endl;
  Sophus::LieGroupCeresTests<Sophus::SE2>(se2_vec, point_vec).testAll();
  return 0;
}
