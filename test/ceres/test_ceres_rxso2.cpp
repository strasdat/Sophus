#include <ceres/ceres.h>
#include <iostream>
#include <fstream>
#include <sophus/rxso2.hpp>

#include "tests.hpp"

template <typename T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

template <>
struct RotationalPart<Sophus::RxSO2d> {
  static double Norm(const typename Sophus::RxSO2d::Tangent &t) {
    return std::abs(t[0]);
  }
};

int main(int, char **) {
  using RxSO2d = Sophus::RxSO2d;
  using Tangent = RxSO2d::Tangent;
  using Point = RxSO2d::Point;
  double const kPi = Sophus::Constants<double>::pi();

  StdVector<RxSO2d> rxso2_vec;
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0.2, 1.)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0.2, 1.1)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0., 1.1)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0.00001, 0.)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0.00001, 0.00001)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(kPi, 0.9)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0.2, 0)) *
                      RxSO2d::exp(Tangent(kPi, 0.0)) *
                      RxSO2d::exp(Tangent(-0.2, 0)));
  rxso2_vec.push_back(RxSO2d::exp(Tangent(0.3, 0)) *
                      RxSO2d::exp(Tangent(kPi, 0.001)) *
                      RxSO2d::exp(Tangent(-0.3, 0)));

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

  std::cerr << "Test Ceres RxSO2" << std::endl;
  Sophus::LieGroupCeresTests<Sophus::RxSO2> test(rxso2_vec, point_vec);
  test.testAll();


  std::shared_ptr<Sophus::BasisSpline<RxSO2d>> so2_spline = test.testSpline(6);
  std::ofstream control("ctrl_pts", std::ofstream::out);
  for (size_t i=0;i<rxso2_vec.size();i++) {
      control << i << " " << rxso2_vec[i].log().transpose() << std::endl;
  }
  control.close();
  std::ofstream inter("inter_pts", std::ofstream::out);
  for (double t=0;t<rxso2_vec.size();t+=0.1) {
      RxSO2d g = so2_spline->parent_T_spline(t);
      inter << t << " " << g.log().transpose() << std::endl;
  }
  inter.close();

  return 0;
}
