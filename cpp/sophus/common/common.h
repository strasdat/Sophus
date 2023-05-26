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
#include <Eigen/Geometry>
#include <farm_ng/core/logging/eigen.h>
#include <farm_ng/core/logging/expected.h>
#include <farm_ng/core/logging/format.h>
#include <farm_ng/core/logging/logger.h>
#include <farm_ng/core/misc/variant_utils.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>

#ifdef __GNUC__
#define TION __PRETTY_FUNCTION__
#elif (_MSC_VER >= 1310)
#define TION __FUNCTION__
#else
#define TION "unknown"
#endif

// from <farm_ng/core/logging/format.h>cd
#define SOPHUS_FORMAT(...) FARM_FORMAT(__VA_ARGS__)

// from <farm_ng/core/logging/logger.h>
#define SOPHUS_ASSERT(...) FARM_ASSERT(__VA_ARGS__)
#define SOPHUS_ASSERT_EQ(...) FARM_ASSERT_EQ(__VA_ARGS__)
#define SOPHUS_ASSERT_GE(...) FARM_ASSERT_GE(__VA_ARGS__)
#define SOPHUS_ASSERT_GT(...) FARM_ASSERT_GT(__VA_ARGS__)
#define SOPHUS_ASSERT_LE(...) FARM_ASSERT_LE(__VA_ARGS__)
#define SOPHUS_ASSERT_LT(...) FARM_ASSERT_LT(__VA_ARGS__)
#define SOPHUS_ASSERT_NE(...) FARM_ASSERT_NE(__VA_ARGS__)
#define SOPHUS_ASSERT_NEAR(...) FARM_ASSERT_NEAR(__VA_ARGS__)
#define SOPHUS_AT(...) FARM_AT(__VA_ARGS__)
#define SOPHUS_INFO(...) FARM_INFO(__VA_ARGS__)
#define SOPHUS_PANIC(...) FARM_PANIC(__VA_ARGS__)
#define SOPHUS_UNIMPLEMENTED(...) FARM_UNIMPLEMENTED(__VA_ARGS__)
#define SOPHUS_UNWRAP(...) FARM_UNWRAP(__VA_ARGS__)

// from <farm_ng/core/logging/expected.h>
#define SOPHUS_TRY(...) FARM_TRY(__VA_ARGS__)
#define SOPHUS_ASSERT_OR_ERROR(...) FARM_ASSERT_OR_ERROR(__VA_ARGS__)
#define SOPHUS_UNEXPECTED(...) FARM_UNEXPECTED(__VA_ARGS__)

namespace sophus {

// from <farm_ng/core/misc/variant_utils.h>
using ::farm_ng::AlwaysFalse;
using ::farm_ng::has_type_v;
using ::farm_ng::Overload;

// from <farm_ng/core/logging/expected.h>
using ::farm_ng::Expected;
using ::farm_ng::Success;

struct UninitTag {};

template <class TScalar>
TScalar const kEpsilon = TScalar(1e-10);

template <>
inline float const kEpsilon<float> = float(1e-5);

static float const kEpsilonF32 = kEpsilon<float>;
static float const kEpsilonF64 = kEpsilon<double>;

template <class TScalar>
TScalar const kEpsilonPlus =
    kEpsilon<TScalar>*(TScalar(1.) + kEpsilon<TScalar>);

using std::sqrt;
template <class TScalar>
TScalar const kEpsilonSqrt = TScalar(1e-5);

template <>
inline float const kEpsilonSqrt<float> = float(3.16227766e-3);

static float const kEpsilonSqrtF32 = kEpsilonSqrt<float>;
static float const kEpsilonSqrtF64 = kEpsilonSqrt<double>;

template <class TScalar>
TScalar const kPi = TScalar(3.141592653589793238462643383279502884);
float const kPiF32 = kPi<float>;
double const kPiF64 = kPi<double>;

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
