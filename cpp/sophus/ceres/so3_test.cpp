// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/ceres/details/test_impl.h"
#include "sophus/lie/se3.h"

#include <ceres/ceres.h>

#include <iostream>

template <class TT>
using StdVector = std::vector<TT, Eigen::aligned_allocator<TT>>;

template <>
struct RotationalPart<sophus::SO3d> {
  static double norm(const typename sophus::SO3d::Tangent &t) {
    return t.norm();
  }
};

int main(int /*unused*/, char ** /*unused*/) {
  using So3Type = sophus::So3<double>;
  using Point = So3Type::Point;
  using Tangent = So3Type::Tangent;
  double const k_pi = sophus::kPi<double>;
  double const epsilon = sophus::kEpsilonF64;

  So3Type c_0 = So3Type::exp(Tangent(0.1, 0.05, -0.7));

  Tangent axis_0(0.18005924, -0.54563405, 0.81845107);

  StdVector<So3Type> so3_vec;
  // Generic tests
  so3_vec.push_back(So3Type::exp(Point(0.2, 0.5, 0.0)));
  so3_vec.push_back(So3Type::exp(Point(0.2, 0.5, -1.0)));
  so3_vec.push_back(So3Type::exp(Point(0., 0., 0.)));
  so3_vec.push_back(So3Type::exp(Point(0., 0., 0.00001)));
  so3_vec.push_back(So3Type::exp(Point(k_pi, 0, 0)));
  so3_vec.push_back(
      So3Type::exp(Point(0.2, 0.5, 0.0)) * So3Type::exp(Point(k_pi, 0, 0)) *
      So3Type::exp(Point(-0.2, -0.5, -0.0)));
  so3_vec.push_back(
      So3Type::exp(Point(0.3, 0.5, 0.1)) * So3Type::exp(Point(k_pi, 0, 0)) *
      So3Type::exp(Point(-0.3, -0.5, -0.1)));

  // Checks if ceres optimization will proceed correctly given problematic
  // or close-to-singular initial conditions, i.e. approx. 180-deg rotation,
  // which trips a flaw in old implementation of So3::log() where the
  // tangent vector's magnitude is set to a constant close to \pi whenever
  // the input rotation's rotation angle is within some tolerance of \pi,
  // giving zero gradients wrt scalar part of quaternion.
  so3_vec.push_back(c_0);
  // Generic rotation angle < pi
  so3_vec.push_back(c_0 * So3Type::exp(axis_0 * 1.0));
  so3_vec.push_back(c_0 * So3Type::exp(axis_0 * -1.0));
  // Generic rotation angle > pi
  so3_vec.push_back(c_0 * So3Type::exp(axis_0 * 4.0));
  so3_vec.push_back(c_0 * So3Type::exp(axis_0 * -4.0));
  // Singular rotation angle = pi
  so3_vec.push_back(c_0 * So3Type::exp(axis_0 * k_pi));
  so3_vec.push_back(c_0 * So3Type::exp(axis_0 * -k_pi));
  so3_vec.push_back(c_0 * So3Type::exp(axis_0 * k_pi * (1.0 + epsilon)));
  so3_vec.push_back(c_0 * So3Type::exp(axis_0 * k_pi * (1.0 - epsilon)));
  so3_vec.push_back(c_0 * So3Type::exp(axis_0 * -k_pi * (1.0 + epsilon)));
  so3_vec.push_back(c_0 * So3Type::exp(axis_0 * -k_pi * (1.0 - epsilon)));

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

  std::cerr << "Test Ceres So3" << std::endl;
  sophus::LieGroupCeresTests<sophus::So3>(so3_vec, point_vec).testAll();
  return 0;
}
