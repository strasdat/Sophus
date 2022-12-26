// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/types.h"

#include <iostream>
#include <sstream>

namespace sophus {
namespace details {

template <class TScalar, class TEnableT = void>
class Pretty {
 public:
  static std::string impl(TScalar s) {
    std::stringstream sstr;
    sstr << s;
    return sstr.str();
  }
};

template <class TPtrT>
class Pretty<TPtrT, std::enable_if_t<std::is_pointer<TPtrT>::value>> {
 public:
  static std::string impl(TPtrT ptr) {
    std::stringstream sstr;
    sstr << std::intptr_t(ptr);
    return sstr.str();
  }
};

template <class TScalar, int kM, int kMatrixDim>
class Pretty<Eigen::Matrix<TScalar, kM, kMatrixDim>, void> {
 public:
  static std::string impl(Eigen::Matrix<TScalar, kM, kMatrixDim> const& v) {
    std::stringstream sstr;
    sstr << "\n" << v << "\n";
    return sstr.str();
  }
};

template <class TT>
std::string pretty(TT const& v) {
  return Pretty<TT>::impl(v);
}

template <class... TArgsT>
void testFailed(
    bool& passed,
    char const* func,
    char const* file,
    int line,
    std::string const& msg) {
  std::fprintf(
      stderr,
      "Test failed in function '%s', "
      "file '%s', line %d.\n",
      func,
      file,
      line);
  std::cerr << msg << "\n\n";
  passed = false;
}

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
}  // namespace sophus

#define SOPHUS_STRINGIFY(x) #x

/// Tests whether condition is true.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST(passed, condition, descr, ...)                     \
  do {                                                                 \
    if (!(condition)) {                                                \
      std::string msg = SOPHUS_FORMAT(                                 \
          "condition ``{}`` is false\n", SOPHUS_STRINGIFY(condition)); \
      msg += SOPHUS_FORMAT(descr, ##__VA_ARGS__);                      \
      sophus::details::testFailed(                                     \
          passed, SOPHUS_FUNCTION, __FILE__, __LINE__, msg);           \
    }                                                                  \
  } while (false)

/// Tests whether left is equal to right given a threshold.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_EQUAL(passed, left, right, descr, ...)   \
  do {                                                       \
    if (left != right) {                                     \
      std::string msg = SOPHUS_FORMAT(                       \
          "{} (={}) is not equal to {} (={})\n",             \
          SOPHUS_STRINGIFY(left),                            \
          sophus::details::pretty(left),                     \
          SOPHUS_STRINGIFY(right),                           \
          sophus::details::pretty(right));                   \
      msg += SOPHUS_FORMAT(descr, ##__VA_ARGS__);            \
      sophus::details::testFailed(                           \
          passed, SOPHUS_FUNCTION, __FILE__, __LINE__, msg); \
    }                                                        \
  } while (false)

/// Tests whether left is equal to right given a threshold.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_NEQ(passed, left, right, descr, ...)     \
  do {                                                       \
    if (left == right) {                                     \
      std::string msg = SOPHUS_FORMAT(                       \
          "{} (={}) should not be equal to {} (={})\n",      \
          SOPHUS_STRINGIFY(left),                            \
          sophus::details::pretty(left),                     \
          SOPHUS_STRINGIFY(right),                           \
          sophus::details::pretty(right));                   \
      msg += SOPHUS_FORMAT(descr, ##__VA_ARGS__);            \
      sophus::details::testFailed(                           \
          passed, SOPHUS_FUNCTION, __FILE__, __LINE__, msg); \
    }                                                        \
  } while (false)

/// Tests whether left is approximately equal to right given a threshold.
/// The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_APPROX(passed, left, right, thr, descr, ...)    \
  do {                                                              \
    auto nrm = sophus::maxMetric((left), (right));                  \
    if (!(nrm < (thr))) {                                           \
      std::string msg = SOPHUS_FORMAT(                              \
          "{} (={}) is not approx {} (={}); {} is {}; nrm is {}\n", \
          SOPHUS_STRINGIFY(left),                                   \
          sophus::details::pretty(left),                            \
          SOPHUS_STRINGIFY(right),                                  \
          sophus::details::pretty(right),                           \
          SOPHUS_STRINGIFY(thr),                                    \
          sophus::details::pretty(thr),                             \
          nrm);                                                     \
      msg += SOPHUS_FORMAT(descr, ##__VA_ARGS__);                   \
      sophus::details::testFailed(                                  \
          passed, SOPHUS_FUNCTION, __FILE__, __LINE__, msg);        \
    }                                                               \
  } while (false)

/// Tests whether left is NOT approximately equal to right given a
/// threshold. The in-out parameter passed will be set to false if test fails.
#define SOPHUS_TEST_NOT_APPROX(passed, left, right, thr, descr, ...)         \
  do {                                                                       \
    auto nrm = sophus::maxMetric((left), (right));                           \
    if (nrm < (thr)) {                                                       \
      std::string msg = SOPHUS_FORMAT(                                       \
          "{} (={}) is approx {} (={}), but it should not!\n {} is {}; nrm " \
          "is {}\n",                                                         \
          SOPHUS_STRINGIFY(left),                                            \
          sophus::details::pretty(left),                                     \
          SOPHUS_STRINGIFY(right),                                           \
          sophus::details::pretty(right),                                    \
          SOPHUS_STRINGIFY(thr),                                             \
          sophus::details::pretty(thr),                                      \
          nrm);                                                              \
      msg += SOPHUS_FORMAT(descr, ##__VA_ARGS__);                            \
      sophus::details::testFailed(                                           \
          passed, SOPHUS_FUNCTION, __FILE__, __LINE__, msg);                 \
    }                                                                        \
  } while (false)
