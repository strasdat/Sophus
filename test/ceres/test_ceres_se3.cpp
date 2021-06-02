#include <ceres/ceres.h>
#include <iostream>
#include <sophus/se3.hpp>

#include "tests.hpp"

template <typename T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

int main(int, char**) {
  using SE3d = Sophus::SE3d;
  using SO3d = Sophus::SO3d;
  using Point = SE3d::Point;
  double const kPi = Sophus::Constants<double>::pi();

  StdVector<SE3d> se3_vec = {
      SE3d(SO3d::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)),
      SE3d(SO3d::exp(Point(0.2, 0.5, -1.0)), Point(10, 0, 0)),
      SE3d(SO3d::exp(Point(0., 0., 0.)), Point(0, 100, 5)),
      SE3d(SO3d::exp(Point(0., 0., 0.00001)), Point(0, 0, 0)),
      SE3d(SO3d::exp(Point(0., 0., 0.00001)),
           Point(0, -0.00000001, 0.0000000001)),
      SE3d(SO3d::exp(Point(0., 0., 0.00001)), Point(0.01, 0, 0)),
      SE3d(SO3d::exp(Point(kPi, 0, 0)), Point(4, -5, 0)),
      SE3d(SO3d::exp(Point(0.2, 0.5, 0.0)), Point(0, 0, 0)) *
          SE3d(SO3d::exp(Point(kPi, 0, 0)), Point(0, 0, 0)) *
          SE3d(SO3d::exp(Point(-0.2, -0.5, -0.0)), Point(0, 0, 0)),
      SE3d(SO3d::exp(Point(0.3, 0.5, 0.1)), Point(2, 0, -7)) *
          SE3d(SO3d::exp(Point(kPi, 0, 0)), Point(0, 0, 0)) *
          SE3d(SO3d::exp(Point(-0.3, -0.5, -0.1)), Point(0, 6, 0))};

  StdVector<Point> point_vec = {
      Point(1.012, 2.73, -1.4), Point(9.2, -7.3, -4.4),  Point(2.5, 0.1, 9.1),
      Point(12.3, 1.9, 3.8),    Point(-3.21, 3.42, 2.3), Point(-8.0, 6.1, -1.1),
      Point(0.0, 2.5, 5.9),     Point(7.1, 7.8, -14),    Point(5.8, 9.2, 0.0),
  };

  std::cerr << "Test Ceres SE3" << std::endl;
  Sophus::LieGroupCeresTests<Sophus::SE3>(se3_vec, point_vec).testAll();
  return 0;
}
