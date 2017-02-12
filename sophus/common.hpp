#ifndef SOPHUS_COMMON_HPP
#define SOPHUS_COMMON_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

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

#define SOPHUS_ENSURE(expr, description, ...) ((void)0)

#elif defined(SOPHUS_ENABLE_ENSURE_HANDLER)

namespace Sophus {
void ensureFailed(char const* function, char const* file, int line,
                  char const* description);
}

#define SOPHUS_ENSURE(expr, description, ...)                               \
  ((expr) ? ((void)0)                                                       \
          : ::Sophus::ensureFailed(                                         \
                SOPHUS_FUNCTION, __FILE__, __LINE__,                        \
                Sophus::details::FormatString((description), ##__VA_ARGS__) \
                    .c_str()))
#else
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
#define SOPHUS_ENSURE(expr, description, ...)                          \
  ((expr) ? ((void)0)                                                  \
          : Sophus::defaultEnsure(SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  (description), ##__VA_ARGS__))
#endif

namespace Sophus {

template <class Scalar>
struct Constants {
  SOPHUS_FUNC static Scalar epsilon() { return Scalar(1e-10); }

  SOPHUS_FUNC static Scalar pi() { return Scalar(M_PI); }
};

template <>
struct Constants<float> {
  SOPHUS_FUNC static float epsilon() { return static_cast<float>(1e-5); }

  SOPHUS_FUNC static float pi() { return static_cast<float>(M_PI); }
};
}

#endif  // SOPHUS_COMMON_HPP
