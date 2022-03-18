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

  StdVector<RxSO3d> rxso3_vec;
  rxso3_vec.push_back(RxSO3d::exp(Tangent(0.2, 0.5, 0.0, 1.)));
  rxso3_vec.push_back(RxSO3d::exp(Tangent(0.2, 0.5, -1.0, 1.1)));
  rxso3_vec.push_back(RxSO3d::exp(Tangent(0., 0., 0., 1.1)));
  rxso3_vec.push_back(RxSO3d::exp(Tangent(0., 0., 0.00001, 0.)));
  rxso3_vec.push_back(RxSO3d::exp(Tangent(0., 0., 0.00001, 0.00001)));
  rxso3_vec.push_back(RxSO3d::exp(Tangent(0., 0., 0.00001, 0)));

  rxso3_vec.push_back(RxSO3d::exp(Tangent(kPi, 0, 0, 0.9)));

  rxso3_vec.push_back(RxSO3d::exp(Tangent(0.2, -0.5, 0, 0)) *
                      RxSO3d::exp(Tangent(kPi, 0, 0, 0)) *
                      RxSO3d::exp(Tangent(-0.2, -0.5, 0, 0)));

  rxso3_vec.push_back(RxSO3d::exp(Tangent(0.3, 0.5, 0.1, 0)) *
                      RxSO3d::exp(Tangent(kPi, 0, 0, 0)) *
                      RxSO3d::exp(Tangent(-0.3, -0.5, -0.1, 0)));

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

  std::cerr << "Test Ceres RxSO2" << std::endl;
  Sophus::LieGroupCeresTests<Sophus::RxSO3>(rxso3_vec, point_vec).testAll();
  return 0;
}
