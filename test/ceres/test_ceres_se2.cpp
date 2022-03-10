#include <ceres/ceres.h>
#include <iostream>
#include <sophus/se2.hpp>

#include "tests.hpp"

template <typename T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

template <>
struct RotationalPart<Sophus::SE2d> {
  static double Norm(const typename Sophus::SE2d::Tangent &t) {
    return std::abs(t[2]);
  }
};

int main(int, char **) {
  using SE2d = Sophus::SE2d;
  using SO2d = Sophus::SO2d;
  using Point = SE2d::Point;
  double const kPi = Sophus::Constants<double>::pi();

  StdVector<SE2d> se2_vec;
  se2_vec.push_back(SE2d(SO2d(0.0), Point(0, 0)));
  se2_vec.push_back(SE2d(SO2d(0.2), Point(10, 0)));
  se2_vec.push_back(SE2d(SO2d(0.), Point(0, 100)));
  se2_vec.push_back(SE2d(SO2d(-1.), Point(20, -1)));
  se2_vec.push_back(SE2d(SO2d(0.00001), Point(-0.00000001, 0.0000000001)));
  se2_vec.push_back(SE2d(SO2d(0.2), Point(0, 0)) *
                    SE2d(SO2d(kPi), Point(0, 0)) *
                    SE2d(SO2d(-0.2), Point(0, 0)));
  se2_vec.push_back(SE2d(SO2d(0.3), Point(2, 0)) *
                    SE2d(SO2d(kPi), Point(0, 0)) *
                    SE2d(SO2d(-0.3), Point(0, 6)));

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

  std::cerr << "Test Ceres SE2" << std::endl;
  Sophus::LieGroupCeresTests<Sophus::SE2>(se2_vec, point_vec).testAll();
  return 0;
}
