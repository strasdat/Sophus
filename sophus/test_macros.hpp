#ifndef SOPUHS_TESTS_MACROS_HPP
#define SOPUHS_TESTS_MACROS_HPP

#include <sophus/types.hpp>

namespace Sophus {
namespace details {

template <class Scalar, class Enable = void>
class Pretty {
 public:
  static std::string impl(Scalar s) { return fmt::format("{}", s); }
};

template <class Ptr>
class Pretty<Ptr, enable_if_t<std::is_pointer<Ptr>::value>> {
 public:
  static std::string impl(Ptr ptr) {
    return fmt::format("{}", std::intptr_t(ptr));
  }
};

template <class Scalar, int M, int N>
class Pretty<Eigen::Matrix<Scalar, M, N>, void> {
 public:
  static std::string impl(Matrix<Scalar, M, N> const& v) {
    return fmt::format("\n{}\n", v);
  }
};

template <class T>
std::string pretty(T const& v) {
  return Pretty<T>::impl(v);
}

template <class... Args>
void testFailed(bool& passed, char const* func, char const* file, int line,
                std::string const& msg) {
  std::cerr << fmt::format("Test failed in function {}, file {}, line {}\n",
                           func, file, line);
  std::cerr << msg << "\n\n";
  passed = false;
}

template <class... Args>
std::string format(char const* text, Args&&... args) {
  return fmt::format(text, std::forward<Args>(args)...);
}

inline std::string format() { return std::string(); }

}  // namespace details

void processTestResult(bool passed) {
  if (!passed) {
    // LCOV_EXCL_START
    std::cerr << "failed!" << std::endl << std::endl;
    exit(-1);
    // LCOV_EXCL_STOP
  }
  std::cerr << "passed." << std::endl << std::endl;
}
}  // namespace Sophus

#define SOPHUS_STRINGIFY(x) #x

/// Tests whether condition is true.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST(passed, condition, ...)                                    \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::string msg = Sophus::details::format("condition ``{}`` is false\n", \
                                                SOPHUS_STRINGIFY(condition));  \
      msg += Sophus::details::format(__VA_ARGS__);                             \
      Sophus::details::testFailed(passed, SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  msg);                                        \
    }                                                                          \
  } while (false)

/// Tests whether left is equal to right given a threshold.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_EQUAL(passed, left, right, ...)                            \
  do {                                                                         \
    if (left != right) {                                                       \
      std::string msg = Sophus::details::format(                               \
          "{} (={}) is not equal to {} (={})\n", SOPHUS_STRINGIFY(left),       \
          Sophus::details::pretty(left), SOPHUS_STRINGIFY(right),              \
          Sophus::details::pretty(right));                                     \
      msg += Sophus::details::format(__VA_ARGS__);                             \
      Sophus::details::testFailed(passed, SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  msg);                                        \
    }                                                                          \
  } while (false)

/// Tests whether left is equal to right given a threshold.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_NEQ(passed, left, right, ...)                              \
  do {                                                                         \
    if (left == right) {                                                       \
      std::string msg = Sophus::details::format(                               \
          "{} (={}) should not be equal to {} (={})\n",                        \
          SOPHUS_STRINGIFY(left), Sophus::details::pretty(left),               \
          SOPHUS_STRINGIFY(right), Sophus::details::pretty(right));            \
      msg += Sophus::details::format(__VA_ARGS__);                             \
      Sophus::details::testFailed(passed, SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  msg);                                        \
    }                                                                          \
  } while (false)

/// Tests whether left is approximately equal to right given a threshold.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_APPROX(passed, left, right, thr, ...)                      \
  do {                                                                         \
    auto nrm = Sophus::maxMetric((left), (right));                             \
    if (!(nrm < (thr))) {                                                      \
      std::string msg = Sophus::details::format(                               \
          "{} (={}) is not approx {} (={}); {} is {}; nrm is {}\n",            \
          SOPHUS_STRINGIFY(left), Sophus::details::pretty(left),               \
          SOPHUS_STRINGIFY(right), Sophus::details::pretty(right),             \
          SOPHUS_STRINGIFY(thr), Sophus::details::pretty(thr), nrm);           \
      msg += Sophus::details::format(__VA_ARGS__);                             \
      Sophus::details::testFailed(passed, SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  msg);                                        \
    }                                                                          \
  } while (false)

/// Tests whether left is NOT approximately equal to right given a
/// threshold. The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_NOT_APPROX(passed, left, right, thr, ...)                  \
  do {                                                                         \
    auto nrm = Sophus::maxMetric((left), (right));                             \
    if (nrm < (thr)) {                                                         \
      std::string msg = Sophus::details::format(                               \
          "{} (={}) is approx {} (={}), but it should not!\n {} is {}; nrm "   \
          "is {}\n",                                                           \
          SOPHUS_STRINGIFY(left), Sophus::details::pretty(left),               \
          SOPHUS_STRINGIFY(right), Sophus::details::pretty(right),             \
          SOPHUS_STRINGIFY(thr), Sophus::details::pretty(thr), nrm);           \
      msg += Sophus::details::format(__VA_ARGS__);                             \
      Sophus::details::testFailed(passed, SOPHUS_FUNCTION, __FILE__, __LINE__, \
                                  msg);                                        \
    }                                                                          \
  } while (false)

#endif  // SOPUHS_TESTS_MACROS_HPP
