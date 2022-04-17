#pragma once

#include "cartesian.hpp"
#include "rxso2.hpp"
#include "rxso3.hpp"
#include "se2.hpp"
#include "se3.hpp"
#include "sim2.hpp"
#include "sim3.hpp"
#include "so2.hpp"
#include "so3.hpp"

namespace Sophus {
namespace interp_details {

template <class Group>
struct Traits;

template <class Scalar, int Dim>
struct Traits<Cartesian<Scalar, Dim>> {
  static bool constexpr supported = true;

  static bool hasShortestPathAmbiguity(Cartesian<Scalar, Dim> const&) {
    return false;
  }
};

template <class Scalar>
struct Traits<SO2<Scalar>> {
  static bool constexpr supported = true;

  static bool hasShortestPathAmbiguity(SO2<Scalar> const& foo_T_bar) {
    using std::abs;
    Scalar angle = abs(foo_T_bar.log());
    Scalar const kPi = Constants<Scalar>::pi();
    return abs(angle - kPi) / (angle + kPi) < Constants<Scalar>::epsilon();
  }
};

template <class Scalar>
struct Traits<RxSO2<Scalar>> {
  static bool constexpr supported = true;

  static bool hasShortestPathAmbiguity(RxSO2<Scalar> const& foo_T_bar) {
    return Traits<SO2<Scalar>>::hasShortestPathAmbiguity(foo_T_bar.so2());
  }
};

template <class Scalar>
struct Traits<SO3<Scalar>> {
  static bool constexpr supported = true;

  static bool hasShortestPathAmbiguity(SO3<Scalar> const& foo_T_bar) {
    using std::abs;
    Scalar angle = abs(foo_T_bar.logAndTheta().theta);
    Scalar const kPi = Constants<Scalar>::pi();
    return abs(angle - kPi) / (angle + kPi) < Constants<Scalar>::epsilon();
  }
};

template <class Scalar>
struct Traits<RxSO3<Scalar>> {
  static bool constexpr supported = true;

  static bool hasShortestPathAmbiguity(RxSO3<Scalar> const& foo_T_bar) {
    return Traits<SO3<Scalar>>::hasShortestPathAmbiguity(foo_T_bar.so3());
  }
};

template <class Scalar>
struct Traits<SE2<Scalar>> {
  static bool constexpr supported = true;

  static bool hasShortestPathAmbiguity(SE2<Scalar> const& foo_T_bar) {
    return Traits<SO2<Scalar>>::hasShortestPathAmbiguity(foo_T_bar.so2());
  }
};

template <class Scalar>
struct Traits<SE3<Scalar>> {
  static bool constexpr supported = true;

  static bool hasShortestPathAmbiguity(SE3<Scalar> const& foo_T_bar) {
    return Traits<SO3<Scalar>>::hasShortestPathAmbiguity(foo_T_bar.so3());
  }
};

template <class Scalar>
struct Traits<Sim2<Scalar>> {
  static bool constexpr supported = true;

  static bool hasShortestPathAmbiguity(Sim2<Scalar> const& foo_T_bar) {
    return Traits<SO2<Scalar>>::hasShortestPathAmbiguity(
        foo_T_bar.rxso2().so2());
    ;
  }
};

template <class Scalar>
struct Traits<Sim3<Scalar>> {
  static bool constexpr supported = true;

  static bool hasShortestPathAmbiguity(Sim3<Scalar> const& foo_T_bar) {
    return Traits<SO3<Scalar>>::hasShortestPathAmbiguity(
        foo_T_bar.rxso3().so3());
    ;
  }
};

}  // namespace interp_details
}  // namespace Sophus
