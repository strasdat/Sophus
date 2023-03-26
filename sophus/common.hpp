/// @file
/// Common functionality.

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>

#include <Eigen/Core>

#undef SOPHUS_COMPILE_TIME_FMT

#ifdef SOPHUS_USE_BASIC_LOGGING

#define SOPHUS_FMT_CSTR(description, ...) description
#define SOPHUS_FMT_STR(description, ...) std::string(description)
#define SOPHUS_FMT_PRINT(description, ...) std::printf("%s\n", description)
#define SOPHUS_FMT_ARG(arg) arg

#else  // !SOPHUS_USE_BASIC_LOGGING

#ifdef __linux__
#define SOPHUS_COMPILE_TIME_FMT
#endif

#ifdef __APPLE__
#include "TargetConditionals.h"
#ifdef TARGET_OS_MAC
#define SOPHUS_COMPILE_TIME_FMT
#endif
#endif

#include <fmt/format.h>
#include <fmt/ostream.h>

#if FMT_VERSION >= 90000
#define SOPHUS_FMT_ARG(arg) fmt::streamed(arg)
#else
#define SOPHUS_FMT_ARG(arg) arg
#endif

#ifdef SOPHUS_COMPILE_TIME_FMT
// To keep compatibility with older libfmt versions,
// disable the compile time check if FMT_STRING is not available.
#ifdef FMT_STRING
// compile-time format check on x
#define SOPHUS_FMT_STRING(x) FMT_STRING(x)
#else
// identity, hence no compile-time check on x
#define SOPHUS_FMT_STRING(x) x
#endif
#else  // ! SOPHUS_COMPILE_TIME_FMT
// identity, hence no compile-time check on x
#define SOPHUS_FMT_STRING(x) x
#endif  // ! SOPHUS_COMPILE_TIME_FMT

#define SOPHUS_FMT_CSTR(description, ...) \
  fmt::format(SOPHUS_FMT_STRING(description), ##__VA_ARGS__).c_str()

#define SOPHUS_FMT_STR(description, ...) \
  fmt::format(SOPHUS_FMT_STRING(description), ##__VA_ARGS__)

#define SOPHUS_FMT_PRINT(description, ...)                   \
  fmt::print(SOPHUS_FMT_STRING(description), ##__VA_ARGS__); \
  fmt::print("\n")

#endif  // !SOPHUS_USE_BASIC_LOGGING

// following boost's assert.hpp
#undef SOPHUS_ENSURE

// ENSURES are similar to ASSERTS, but they are always checked for (including in
// release builds). At the moment there are no ASSERTS in Sophus which should
// only be used for checks which are performance critical.

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

// NVCC on windows has issues with defaulting the Map specialization
// constructors, so special case that specific platform case.
#if defined(_MSC_VER) && defined(__CUDACC__)
#define SOPHUS_WINDOW_NVCC_FALLBACK
#endif

#define SOPHUS_FUNC EIGEN_DEVICE_FUNC

#if defined(SOPHUS_DISABLE_ENSURES)

#define SOPHUS_ENSURE(expr, ...) ((void)0)

#elif defined(SOPHUS_ENABLE_ENSURE_HANDLER)

namespace Sophus {
void ensureFailed(char const* function, char const* file, int line,
                  char const* description);
}

#define SOPHUS_ENSURE(expr, description, ...)                        \
  ((expr)                                                            \
       ? ((void)0)                                                   \
       : ::Sophus::ensureFailed(SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                SOPHUS_FMT_CSTR(description, ##__VA_ARGS__)))
#else

#define SOPHUS_DEDAULT_ENSURE_FAILURE_IMPL(function, file, line, description, \
                                           ...)                               \
  do {                                                                        \
    std::printf(                                                              \
        "Sophus ensure failed in function '%s', "                             \
        "file '%s', line %d.\n",                                              \
        function, file, line);                                                \
    SOPHUS_FMT_PRINT(description, ##__VA_ARGS__);                             \
    std::abort();                                                             \
  } while (false)

#ifdef __CUDACC__
#define SOPHUS_ENSURE(expr, description, ...)                                  \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::printf(                                                             \
          "Sophus ensure failed in function '%s', file '%s', line %d.\n",      \
          SOPHUS_FUNCTION, __FILE__, __LINE__);                                \
      std::printf("%s", description);                                          \
      /* there is no std::abort in cuda kernels, hence we just print the error \
       * message here*/                                                        \
    }                                                                          \
  } while (false)
#else
#define SOPHUS_ENSURE(expr, ...)                                              \
  do {                                                                        \
    if (!(expr)) {                                                            \
      SOPHUS_DEDAULT_ENSURE_FAILURE_IMPL(SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                         ##__VA_ARGS__);                      \
    }                                                                         \
  } while (false)
#endif
#endif

namespace Sophus {

template <class Scalar>
struct Constants {
  SOPHUS_FUNC static Scalar epsilon() { return Scalar(1e-10); }

  SOPHUS_FUNC static Scalar epsilonPlus() {
    return epsilon() * (Scalar(1.) + epsilon());
  }

  SOPHUS_FUNC static Scalar epsilonSqrt() {
    using std::sqrt;
    return sqrt(epsilon());
  }

  SOPHUS_FUNC static Scalar pi() {
    return Scalar(3.141592653589793238462643383279502884);
  }
};

template <>
struct Constants<float> {
  SOPHUS_FUNC static float constexpr epsilon() {
    return static_cast<float>(1e-5);
  }
  SOPHUS_FUNC static float epsilonPlus() {
    return epsilon() * (1.f + epsilon());
  }

  SOPHUS_FUNC static float epsilonSqrt() { return std::sqrt(epsilon()); }

  SOPHUS_FUNC static float constexpr pi() {
    return 3.141592653589793238462643383279502884f;
  }
};

/// Nullopt type of lightweight optional class.
struct nullopt_t {
  explicit constexpr nullopt_t() {}
};

constexpr nullopt_t nullopt{};

/// Lightweight optional implementation which requires ``T`` to have a
/// default constructor.
///
/// TODO: Replace with std::optional once Sophus moves to c++17.
///
template <class T>
class optional {
 public:
  optional() : is_valid_(false) {}

  optional(nullopt_t) : is_valid_(false) {}

  optional(T const& type) : type_(type), is_valid_(true) {}

  explicit operator bool() const { return is_valid_; }

  T const* operator->() const {
    SOPHUS_ENSURE(is_valid_, "must be valid");
    return &type_;
  }

  T* operator->() {
    SOPHUS_ENSURE(is_valid_, "must be valid");
    return &type_;
  }

  T const& operator*() const {
    SOPHUS_ENSURE(is_valid_, "must be valid");
    return type_;
  }

  T& operator*() {
    SOPHUS_ENSURE(is_valid_, "must be valid");
    return type_;
  }

 private:
  T type_;
  bool is_valid_;
};

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <class G>
struct IsUniformRandomBitGenerator {
  static const bool value = std::is_unsigned<typename G::result_type>::value &&
                            std::is_unsigned<decltype(G::min())>::value &&
                            std::is_unsigned<decltype(G::max())>::value;
};
}  // namespace Sophus
