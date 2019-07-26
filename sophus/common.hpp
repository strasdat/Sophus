/// @file
/// Common functionality.

#ifndef SOPHUS_COMMON_HPP
#define SOPHUS_COMMON_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <type_traits>

#include <Eigen/Core>

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

#define SOPHUS_FUNC EIGEN_DEVICE_FUNC

#if defined(__GNUC__) || defined(__clang__)
#define SOPHUS_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define SOPHUS_DEPRECATED __declspec(deprecated)
#else
#define SOPHUS_DEPRECATED
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

inline void FormatStream(std::stringstream& stream, char const* text) {
  stream << text;
  return;
}

// Following: http://en.cppreference.com/w/cpp/language/parameter_pack
template <class T, typename... Args>
void FormatStream(std::stringstream& stream, char const* text, T&& arg,
                  Args&&... args) {
  static_assert(IsStreamable<T>::value,
                "One of the args has no ostream overload!");
  for (; *text != '\0'; ++text) {
    if (*text == '%') {
      ArgToStream<T&&>::impl(stream, std::forward<T>(arg));
      FormatStream(stream, text + 1, std::forward<Args>(args)...);
      return;
    }
    stream << *text;
  }
  stream << "\nFormat-Warning: There are " << sizeof...(Args) + 1
         << " args unused.";
  return;
}

template <class... Args>
std::string FormatString(char const* text, Args&&... args) {
  std::stringstream stream;
  FormatStream(stream, text, std::forward<Args>(args)...);
  return stream.str();
}

inline std::string FormatString() { return std::string(); }
}  // namespace details
}  // namespace Sophus

#if defined(SOPHUS_DISABLE_ENSURES)

#define SOPHUS_ENSURE(expr, ...) ((void)0)

#elif defined(SOPHUS_ENABLE_ENSURE_HANDLER)

namespace Sophus {
void ensureFailed(char const* function, char const* file, int line,
                  char const* description);
}

#define SOPHUS_ENSURE(expr, ...)                     \
  ((expr) ? ((void)0)                                \
          : ::Sophus::ensureFailed(                  \
                SOPHUS_FUNCTION, __FILE__, __LINE__, \
                Sophus::details::FormatString(__VA_ARGS__).c_str()))
#else
// LCOV_EXCL_START

namespace Sophus {
template <class... Args>
SOPHUS_FUNC void defaultEnsure(char const* function, char const* file, int line,
                               char const* description, Args&&... args) {
  std::printf("Sophus ensure failed in function '%s', file '%s', line %d.\n",
              function, file, line);
#ifdef __CUDACC__
  std::printf("%s", description);
#else
  std::cout << details::FormatString(description, std::forward<Args>(args)...)
            << std::endl;
  std::abort();
#endif
}
}  // namespace Sophus

// LCOV_EXCL_STOP
#define SOPHUS_ENSURE(expr, ...)                                       \
  ((expr) ? ((void)0)                                                  \
          : Sophus::defaultEnsure(SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  ##__VA_ARGS__))
#endif

namespace Sophus {

template <class Scalar>
struct Constants {
  SOPHUS_FUNC static Scalar epsilon() { return Scalar(1e-10); }

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

  SOPHUS_FUNC static float epsilonSqrt() { return std::sqrt(epsilon()); }

  SOPHUS_FUNC static float constexpr pi() {
    return 3.141592653589793238462643383279502884f;
  }
};

/// Lightweight expected implementation which requires type T and error E to
/// have default constructors.
///
/// In a nutshell, this is an either/or type: It either contains a value of type
/// T, or it contains an error of type E.
///
template <class T, class E>
class Expected {
 public:
  Expected(E error) : type_(), error_(std::move(error)), has_value_(false) {}

  Expected(T type) : type_(type), error_(), has_value_(true) {}

  bool has_value() const { return has_value_; }
  explicit operator bool() const { return has_value(); }

  /// Returns pointer to value of type T.
  ///
  /// Precondition: has_value() must be true.
  ///
  T const* operator->() const {
    SOPHUS_ENSURE(has_value_, "must be valid");
    return &type_;
  }

  T* operator->() {
    SOPHUS_ENSURE(has_value_, "must be valid");
    return &type_;
  }

  /// Returns value of type T.
  ///
  /// Precondition: has_value() must be true.
  ///
  T const& operator*() const {
    SOPHUS_ENSURE(has_value_, "must be valid");
    return type_;
  }

  T& operator*() {
    SOPHUS_ENSURE(has_value_, "must be valid");
    return type_;
  }

  /// Returns error of type E.
  ///
  /// Precondition: has_value() must be false.
  ///
  const E& error() const {
    SOPHUS_ENSURE(!has_value_, "must not be valid");
    return error_;
  }
  E& error() {
    SOPHUS_ENSURE(!has_value_, "must not be valid");
    return error_;
  }

 private:
  T type_;
  E error_;
  bool has_value_;
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

#endif  // SOPHUS_COMMON_HPP
