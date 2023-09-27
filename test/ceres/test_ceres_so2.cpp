#include <ceres/ceres.h>
#include <iostream>
#include <fstream>
#include <sophus/so2.hpp>

#include "tests.hpp"

template <typename T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

template <>
struct RotationalPart<Sophus::SO2d> {
  static double Norm(const typename Sophus::SO2d::Tangent &t) {
    return std::abs(t);
  }
};

int main(int, char **) {
  using SO2d = Sophus::SO2d;
  using Point = SO2d::Point;
  double const kPi = Sophus::Constants<double>::pi();

  StdVector<SO2d> so2_vec;
  so2_vec.emplace_back(SO2d::exp(0.0));
  so2_vec.emplace_back(SO2d::exp(0.2));
  so2_vec.emplace_back(SO2d::exp(10.));
  so2_vec.emplace_back(SO2d::exp(0.00001));
  so2_vec.emplace_back(SO2d::exp(kPi));
  so2_vec.emplace_back(SO2d::exp(0.2) * SO2d::exp(kPi) * SO2d::exp(-0.2));
  so2_vec.emplace_back(SO2d::exp(-0.3) * SO2d::exp(kPi) * SO2d::exp(0.3));

  StdVector<Point> point_vec;
  point_vec.emplace_back(Point(1.012, 2.73));
  point_vec.emplace_back(Point(9.2, -7.3));
  point_vec.emplace_back(Point(2.5, 0.1));
  point_vec.emplace_back(Point(12.3, 1.9));
  point_vec.emplace_back(Point(-3.21, 3.42));
  point_vec.emplace_back(Point(-8.0, 6.1));
  point_vec.emplace_back(Point(0.0, 2.5));
  point_vec.emplace_back(Point(7.1, 7.8));
  point_vec.emplace_back(Point(5.8, 9.2));

  std::cerr << "Test Ceres SO2" << std::endl;
  Sophus::LieGroupCeresTests<Sophus::SO2> test(so2_vec, point_vec);
  test.testAll();

#if 0
  // Example code to plot the interpolated spline
  std::shared_ptr<Sophus::BasisSpline<SO2d>> so2_spline = test.testSpline(6);
  std::ofstream control("ctrl_pts", std::ofstream::out);
  for (size_t i=0;i<so2_vec.size();i++) {
      control << i << " " << so2_vec[i].log() << std::endl;
  }
  control.close();
  std::ofstream inter("inter_pts", std::ofstream::out);
  for (double t=0;t<so2_vec.size();t+=0.1) {
      SO2d g = so2_spline->parent_T_spline(t);
      inter << t << " " << g.log() << std::endl;
  }
  inter.close();
#endif

  return 0;
}
