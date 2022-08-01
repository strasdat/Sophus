#include <ceres/ceres.h>
#include <iostream>
#include <sophus/sim2.hpp>

#include "tests.hpp"

template <typename T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

template <>
struct RotationalPart<Sophus::Sim2d> {
  static double Norm(const typename Sophus::Sim2d::Tangent &t) {
    return std::abs(t[2]);
  }
};

int main(int, char **) {
  using Sim2d = Sophus::Sim2d;
  using RxSO2d = Sophus::RxSO2d;
  using Point = Sim2d::Point;
  using Vector2d = Sophus::Vector2d;
  double const kPi = Sophus::Constants<double>::pi();

  StdVector<Sim2d> sim2_vec;

  sim2_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0.2, 1.)), Point(0, 0)));

  sim2_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0.2, 1.1)), Point(10, 0)));

  sim2_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0., 0.)), Point(0, 10)));

  sim2_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0.00001, 0.)), Point(0, 0)));

  sim2_vec.push_back(
      Sim2d(RxSO2d::exp(Vector2d(0.00001, 0.0000001)), Point(1, -1.00000001)));

  sim2_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0., 0.)), Point(0.01, 0)));

  sim2_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(kPi, 0.9)), Point(4, 0)));

  sim2_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0.2, 0)), Point(0, 0)) *
                     Sim2d(RxSO2d::exp(Vector2d(kPi, 0)), Point(0, 0)) *
                     Sim2d(RxSO2d::exp(Vector2d(-0.2, 0)), Point(0, 0)));

  sim2_vec.push_back(Sim2d(RxSO2d::exp(Vector2d(0.3, 0)), Point(2, -7)) *
                     Sim2d(RxSO2d::exp(Vector2d(kPi, 0)), Point(0, 0)) *
                     Sim2d(RxSO2d::exp(Vector2d(-0.3, 0)), Point(0, 6)));

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

  std::cerr << "Test Ceres Sim2" << std::endl;
  Sophus::LieGroupCeresTests<Sophus::Sim2>(sim2_vec, point_vec).testAll();
  return 0;
}
