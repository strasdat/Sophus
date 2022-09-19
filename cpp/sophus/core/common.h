// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Common functionality.

#pragma once

#include <Eigen/Core>
#include <farm_ng/core/logging/logger.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>

#ifdef __GNUC__
#define SOPHUS_FUNCTION __PRETTY_FUNCTION__
#elif (_MSC_VER >= 1310)
#define SOPHUS_FUNCTION __FUNCTION__
#else
#define SOPHUS_FUNCTION "unknown"
#endif

// Make sure this compiles with older versions of Eigen which do not have
// EIGEN_DEVICE_FUNC defined.
#ifndef EIGEN_DEVICE_FUNC
#define EIGEN_DEVICE_FUNC
#endif

#define SOPHUS_FUNC EIGEN_DEVICE_FUNC

namespace sophus {

template <class ScalarT>
ScalarT const kEpsilon = ScalarT(1e-10);

template <>
inline float const kEpsilon<float> = float(1e-5);

const float kEpsilonF32 = kEpsilon<float>;
const float kEpsilonF64 = kEpsilon<double>;

template <class ScalarT>
ScalarT const kEpsilonPlus =
    kEpsilon<ScalarT>*(ScalarT(1.) + kEpsilon<ScalarT>);

using std::sqrt;
template <class ScalarT>
ScalarT const kEpsilonSqrt = sqrt(kEpsilon<ScalarT>);

const float kEpsilonSqrtF32 = kEpsilonSqrt<float>;
const float kEpsilonSqrtF64 = kEpsilonSqrt<double>;

template <class ScalarT>
ScalarT const kPi = ScalarT(3.141592653589793238462643383279502884);
const float kPiF32 = kPi<float>;
const float kPiF64 = kPi<double>;

template <class GeneratorT>
struct IsUniformRandomBitGenerator {
  static const bool kValue =
      std::is_unsigned<typename GeneratorT::result_type>::value &&
      std::is_unsigned<decltype(GeneratorT::min())>::value &&
      std::is_unsigned<decltype(GeneratorT::max())>::value;
};

template <class GeneratorT>
constexpr bool kIsUniformRandomBitGeneratorV =
    IsUniformRandomBitGenerator<GeneratorT>::kValue;
}  // namespace sophus
