#include <ceres/ceres.h>
#include <iostream>
#include <sophus/sim2.hpp>

#include "tests.hpp"

template <typename T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

int main(int, char**) {
  using Sim2d = Sophus::Sim2d;
  using RxSO2d = Sophus::RxSO2d;
  using Point = Sim2d::Point;
  using Vector2d = Sophus::Vector2d;
  double const kPi = Sophus::Constants<double>::pi();

  StdVector<Sim2d> sim2_vec = {

      Sim2d(RxSO2d::exp(Vector2d(0.2, 1.)), Point(0, 0)),

      Sim2d(RxSO2d::exp(Vector2d(0.2, 1.1)), Point(10, 0)),

      Sim2d(RxSO2d::exp(Vector2d(0., 0.)), Point(0, 10)),

      Sim2d(RxSO2d::exp(Vector2d(0.00001, 0.)), Point(0, 0)),

      Sim2d(RxSO2d::exp(Vector2d(0.00001, 0.0000001)), Point(1, -1.00000001)),

      Sim2d(RxSO2d::exp(Vector2d(0., 0.)), Point(0.01, 0)),

      Sim2d(RxSO2d::exp(Vector2d(kPi, 0.9)), Point(4, 0)),

      Sim2d(RxSO2d::exp(Vector2d(0.2, 0)), Point(0, 0)) *
          Sim2d(RxSO2d::exp(Vector2d(kPi, 0)), Point(0, 0)) *
          Sim2d(RxSO2d::exp(Vector2d(-0.2, 0)), Point(0, 0)),

      Sim2d(RxSO2d::exp(Vector2d(0.3, 0)), Point(2, -7)) *
          Sim2d(RxSO2d::exp(Vector2d(kPi, 0)), Point(0, 0)) *
          Sim2d(RxSO2d::exp(Vector2d(-0.3, 0)), Point(0, 6)),
  };

  StdVector<Point> point_vec = {
      Point(1.012, 2.73), Point(9.2, -7.3),   Point(2.5, 0.1),
      Point(12.3, 1.9),   Point(-3.21, 3.42), Point(-8.0, 6.1),
      Point(0.0, 2.5),    Point(7.1, 7.8),    Point(5.8, 9.2)};

  std::cerr << "Test Ceres Sim2" << std::endl;
  Sophus::LieGroupCeresTests<Sophus::Sim2>(sim2_vec, point_vec).testAll();
  return 0;
}
