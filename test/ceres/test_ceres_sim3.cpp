#include <ceres/ceres.h>
#include <iostream>
#include <sophus/sim3.hpp>

#include "tests.hpp"

template <typename T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

int main(int, char**) {
  using RxSO3d = Sophus::RxSO3d;
  using Sim3d = Sophus::Sim3d;
  using Point = Sim3d::Point;
  using Vector4d = Eigen::Vector4d;
  double const kPi = Sophus::Constants<double>::pi();

  StdVector<Sim3d> sim3_vec = {
      Sim3d(RxSO3d::exp(Vector4d(0.2, 0.5, 0.0, 1.)), Point(0, 0, 0)),
      Sim3d(RxSO3d::exp(Vector4d(0.2, 0.5, -1.0, 1.1)), Point(10, 0, 0)),
      Sim3d(RxSO3d::exp(Vector4d(0., 0., 0., 0.001)), Point(0, 10, 5)),
      Sim3d(RxSO3d::exp(Vector4d(0., 0., 0., 1.1)), Point(0, 10, 5)),
      Sim3d(RxSO3d::exp(Vector4d(0., 0., 0.00001, 0.)), Point(0, 0, 0)),
      Sim3d(RxSO3d::exp(Vector4d(0., 0., 0.00001, 0.0000001)),
            Point(1, -1.00000001, 2.0000000001)),
      Sim3d(RxSO3d::exp(Vector4d(0., 0., 0.00001, 0)), Point(0.01, 0, 0)),
      Sim3d(RxSO3d::exp(Vector4d(kPi, 0, 0, 0.9)), Point(4, -5, 0)),
      Sim3d(RxSO3d::exp(Vector4d(0.2, 0.5, 0.0, 0)), Point(0, 0, 0)) *
          Sim3d(RxSO3d::exp(Vector4d(kPi, 0, 0, 0)), Point(0, 0, 0)) *
          Sim3d(RxSO3d::exp(Vector4d(-0.2, -0.5, -0.0, 0)), Point(0, 0, 0)),
      Sim3d(RxSO3d::exp(Vector4d(0.3, 0.5, 0.1, 0)), Point(2, 0, -7)) *
          Sim3d(RxSO3d::exp(Vector4d(kPi, 0, 0, 0)), Point(0, 0, 0)) *
          Sim3d(RxSO3d::exp(Vector4d(-0.3, -0.5, -0.1, 0)), Point(0, 6, 0))};

  StdVector<Point> point_vec = {
      Point(1.012, 2.73, -1.4), Point(9.2, -7.3, -4.4),  Point(2.5, 0.1, 9.1),
      Point(12.3, 1.9, 3.8),    Point(-3.21, 3.42, 2.3), Point(-8.0, 6.1, -1.1),
      Point(0.0, 2.5, 5.9),     Point(7.1, 7.8, -14),    Point(5.8, 9.2, 0.0),
  };

  std::cerr << "Test Ceres Sim3" << std::endl;
  Sophus::LieGroupCeresTests<Sophus::Sim3>(sim3_vec, point_vec).testAll();
  return 0;
}
