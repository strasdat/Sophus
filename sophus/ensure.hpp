// This file is part of Sophus.
//
// Copyright 2013 Hauke Strasdat
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef SOPHUS_ENSURE_HPP
#define SOPHUS_ENSURE_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <Eigen/Core>

#ifdef SOPHUS_CERES_FOUND
#include <ceres/ceres.h>
#endif

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
template <typename T>
class IsStreamable {
 private:
  template <typename TT>
  static auto test(int)
      -> decltype(std::declval<std::stringstream&>() << std::declval<TT>(),
                  std::true_type());

  template <typename>
  static auto test(...) -> std::false_type;

 public:
  static const bool value = decltype(test<T>(0))::value;
};

template <typename T>
class ArgToStream {
 public:
  static void impl(std::stringstream& stream, T arg) { stream << arg; }
};

#ifdef SOPHUS_CERES_FOUND
// Hack to side-step broken ostream overloads of Eigen types with Jet Scalars.
template <int N, int Rows, int Cols, int Opts, int MaxRows, int MaxCols>
class ArgToStream<Eigen::Transpose<
    Eigen::Matrix<ceres::Jet<double, N>, Rows, Cols, Opts, MaxRows, MaxCols>>> {
 public:
  static void impl(
      std::stringstream& stream,
      Eigen::Transpose<Eigen::Matrix<ceres::Jet<double, N>, Rows, Cols, Opts,
                                     MaxRows, MaxCols>>) {
    stream << "[jet]";
  }
};

template <int N, int Rows, int Cols, int Opts, int MaxRows, int MaxCols>
class ArgToStream<Eigen::Transpose<const Eigen::Matrix<
    ceres::Jet<double, N>, Rows, Cols, Opts, MaxRows, MaxCols>>> {
 public:
  static void impl(
      std::stringstream& stream,
      Eigen::Transpose<const Eigen::Matrix<ceres::Jet<double, N>, Rows, Cols,
                                           Opts, MaxRows, MaxCols>>) {
    stream << "[jet]";
  }
};
#endif  // SOPHUS_CERES_FOUND

inline void FormatStream(std::stringstream& stream, const char* text) {
  stream << text;
  return;
}

// Following: http://en.cppreference.com/w/cpp/language/parameter_pack
template <typename T, typename... Args>
void FormatStream(std::stringstream& stream, const char* text, T arg,
                  Args... args) {
  static_assert(IsStreamable<T>::value,
                "One of the args has not ostream overload!");
  for (; *text != '\0'; ++text) {
    if (*text == '%') {
      ArgToStream<T>::impl(stream, arg);
      FormatStream(stream, text + 1, args...);
      return;
    }
    stream << *text;
  }
  stream << "\nFormat-Warning: There are " << sizeof...(Args) + 1
         << " args unused.";
  return;
}

template <typename... Args>
std::string FormatString(const char* text, Args... args) {
  std::stringstream stream;
  FormatStream(stream, text, args...);
  return stream.str();
}

inline std::string FormatString() {
    return std::string();
}
}  // namespace details
}  // namespace Sophus

#if defined(SOPHUS_DISABLE_ENSURES)

#define SOPHUS_ENSURE(expr, description, ...) ((void)0)

#elif defined(SOPHUS_ENABLE_ENSURE_HANDLER)

namespace Sophus {
void ensureFailed(const char* function, const char* file, int line,
                  const char* description);
}

#define SOPHUS_ENSURE(expr, description, ...)                               \
  ((expr) ? ((void)0)                                                       \
          : ::Sophus::ensureFailed(                                         \
                SOPHUS_FUNCTION, __FILE__, __LINE__,                        \
                Sophus::details::FormatString((description), ##__VA_ARGS__) \
                    .c_str()))
#else
namespace Sophus {
template <typename... Args>
SOPHUS_FUNC void defaultEnsure(const char* function, const char* file, int line,
                               const char* description, Args&&... args) {
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

#endif  // SOPHUS_ENSURE_HPP
