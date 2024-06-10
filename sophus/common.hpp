/// @file
/// Common functionality.

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <type_traits>

#include <Eigen/Core>

#ifdef __GNUC__
#define SOPHUS_FUNCTION __PRETTY_FUNCTION__
#elif (_MSC_VER >= 1310)
#define SOPHUS_FUNCTION __FUNCTION__
#else
#define SOPHUS_FUNCTION "unknown"
#endif

namespace Sophus {
namespace details {

// Following: http://stackoverflow.com/a/22759544
template <class T>
class IsStreamable {
 private:
  template <class TT>
  static auto test(int)
      -> decltype(std::declval<std::stringstream&>() << std::declval<TT>(),
                  std::true_type());

  template <class>
  static auto test(...) -> std::false_type;

 public:
  static bool const value = decltype(test<T>(0))::value;
};

template <class T>
class ArgToStream {
 public:
  static void impl(std::stringstream& stream, T&& arg) {
    stream << std::forward<T>(arg);
  }
};

inline void formatStream(std::stringstream& stream, char const* text) {
  stream << text;
  return;
}

// Following: http://en.cppreference.com/w/cpp/language/parameter_pack
template <class T, typename... Args>
void formatStream(std::stringstream& stream, char const* text, T&& arg,
                  Args&&... args) {
  static_assert(IsStreamable<T>::value,
                "One of the args has no ostream overload!");
  for (; *text != '\0'; ++text) {
    if (*text == '{' && *(text + 1) == '}') {
      ArgToStream<T&&>::impl(stream, std::forward<T>(arg));
      formatStream(stream, text + 2, std::forward<Args>(args)...);
      return;
    }
    stream << *text;
  }
  stream << "\nFormat-Warning: There are " << sizeof...(Args) + 1
         << " args unused.";
  return;
}

template <class... Args>
std::string formatString(char const* text, Args&&... args) {
  std::stringstream stream;
  formatStream(stream, text, std::forward<Args>(args)...);
  return stream.str();
}

inline std::string formatString() { return std::string(); }
}  // namespace details
}  // namespace Sophus

#define SOPHUS_DETAILS_FMT_STR(description, ...) \
  ::Sophus::details::formatString(description, ##__VA_ARGS__)

#define SOPHUS_DETAILS_FMT_PRINT(description, ...) \
  std::printf(                                     \
      "%s\n",                                      \
      ::Sophus::details::formatString(description, ##__VA_ARGS__).c_str())

#define SOPHUS_DETAILS_FMT_LOG(description, ...) \
  std::printf(                                   \
      "[%s:%d] %s\n", __FILE__, __LINE__,        \
      ::Sophus::details::formatString(description, ##__VA_ARGS__).c_str());

// FARM_ENSURE (aka runtime asserts). There are two modes:
//
// 1. SOPHUS_ENABLE_ENSURE_HANDLER: custom ensure handle
// 2. default mode: log on ENSURE message and panic
//
#if defined(SOPHUS_ENABLE_ENSURE_HANDLER)
// 1. custom ensure handle, e.g., throw an exception
//
// One needs to link against a custom ensure handler.

namespace Sophus {
void ensureFailed(char const* function, char const* file, int line,
                  char const* description);
}

#define SOPHUS_ENSURE_FAILED(description, ...) \
  ::Sophus::ensureFailed(                      \
      SOPHUS_FUNCTION, __FILE__, __LINE__,     \
      ::Sophus::details::formatString(description, ##__VA_ARGS__).c_str())
#else  // 1. custom ensure handle
// 2. default mode: log on ENSURE message and panic
#define SOPHUS_ENSURE_FAILED(description, ...)            \
  do {                                                    \
    SOPHUS_DETAILS_FMT_LOG("SOPHUS_ENSURE failed:");      \
    SOPHUS_DETAILS_FMT_PRINT(description, ##__VA_ARGS__); \
    std::abort();                                         \
  } while (false)

#endif  // 2. default mode

#ifdef __CUDACC__
#define SOPHUS_ENSURE(expr, description, ...)                                  \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::printf("Sophus ensure failed in file '%s', line %d.\n", __FILE__,   \
                  __LINE__);                                                   \
      std::printf("%s", description);                                          \
      /* there is no std::abort in cuda kernels, hence we just print the error \
       * message here*/                                                        \
    }                                                                          \
  } while (false)
#else
#define SOPHUS_ENSURE(expr, description, ...)           \
  do {                                                  \
    if (!(expr)) {                                      \
      SOPHUS_ENSURE_FAILED(description, ##__VA_ARGS__); \
    }                                                   \
  } while (false)
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

template <class G>
struct IsUniformRandomBitGenerator {
  static const bool value = std::is_unsigned<typename G::result_type>::value &&
                            std::is_unsigned<decltype(G::min())>::value &&
                            std::is_unsigned<decltype(G::max())>::value;
};

}  // namespace Sophus
