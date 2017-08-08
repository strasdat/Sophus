#ifndef SOPHUS_NUM_DIFF_HPP
#define SOPHUS_NUM_DIFF_HPP

#include <type_traits>
#include <utility>

#include "types.hpp"

namespace Sophus {

// Numerical differentiation using finite differences
template <class Scalar>
class NumDiff {
 public:
  // Calculates the derivative of a curve at a point ``t``.
  //
  // Here, a curve is a function from a Scalar to a Euclidean space. Thus, it
  // returns either a Scalar, a vector or a matrix.
  //
  template <class Fn>
  static auto curve(Fn const& curve, Scalar t,
                    Scalar h = Constants<Scalar>::epsilon)
      -> decltype(curve(t)) {
    using ReturnType = decltype(curve(t));
    static_assert(IsFloatingPoint<ReturnType>::value,
                  "ReturnType must be either a floating point scalar, "
                  "vector or matrix.");
    static_assert(
        std::is_same<typename GetScalar<ReturnType>::Scalar, Scalar>::value,
        "Input and output scalar must be same type (e.g. both float, or both "
        "double).");
    return (curve(t + h) - curve(t - h)) / (Scalar(2) * h);
  }
};
}  // namespace Sophus

#endif  // SOPHUS_NUM_DIFF_HPP
