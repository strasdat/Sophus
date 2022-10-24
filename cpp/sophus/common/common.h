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
#include <farm_ng/core/logging/eigen.h>
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

template <class TScalar>
TScalar const kEpsilon = TScalar(1e-10);

template <>
inline float const kEpsilon<float> = float(1e-5);

float const kEpsilonF32 = kEpsilon<float>;
float const kEpsilonF64 = kEpsilon<double>;

template <class TScalar>
TScalar const kEpsilonPlus =
    kEpsilon<TScalar>*(TScalar(1.) + kEpsilon<TScalar>);

using std::sqrt;
template <class TScalar>
TScalar const kEpsilonSqrt = sqrt(kEpsilon<TScalar>);

float const kEpsilonSqrtF32 = kEpsilonSqrt<float>;
float const kEpsilonSqrtF64 = kEpsilonSqrt<double>;

template <class TScalar>
TScalar const kPi = TScalar(3.141592653589793238462643383279502884);
float const kPiF32 = kPi<float>;
float const kPiF64 = kPi<double>;

template <class TGenerator>
struct IsUniformRandomBitGenerator {
  static bool const kValue =
      std::is_unsigned<typename TGenerator::result_type>::value &&
      std::is_unsigned<decltype(TGenerator::min())>::value &&
      std::is_unsigned<decltype(TGenerator::max())>::value;
};

template <class TGenerator>
bool constexpr kIsUniformRandomBitGeneratorV =
    IsUniformRandomBitGenerator<TGenerator>::kValue;
}  // namespace sophus
