
// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "point_concepts.h"

#include <Eigen/Core>
#include <sophus/lie/se3.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace sophus {

namespace details {

// EigenDenseType may be a map or view or abstract base class or something.
// This is an alias for the corresponding concrete type with storage
template <EigenDenseType TPoint>
using EigenConcreteType = std::decay_t<decltype(std::declval<TPoint>().eval())>;

template <class TScalar>
class Square;

template <ScalarType TScalar>
class Square<TScalar> {
 public:
  static TScalar impl(TScalar const& v) { return v * v; }
};

template <EigenDenseType TPoint>
class Square<TPoint> {
 public:
  static auto impl(TPoint const& v) -> typename TPoint::Scalar {
    return v.squaredNorm();
  }
};

template <class TScalar>
class Min;

template <ScalarType TScalar>
class Min<TScalar> {
 public:
  static TScalar impl(TScalar const& lhs, TScalar const& rhs) {
    return std::min(lhs, rhs);
  }
};

template <EigenDenseType TPoint>
class Min<TPoint> {
 public:
  static EigenConcreteType<TPoint> impl(TPoint const& lhs, TPoint const& rhs) {
    return lhs.cwiseMin(rhs);
  }
};

template <class TScalar>
class Max;

template <ScalarType TScalar>
class Max<TScalar> {
 public:
  static TScalar impl(TScalar const& lhs, TScalar const& rhs) {
    return std::max(lhs, rhs);
  }
};

template <EigenDenseType TPoint>
class Max<TPoint> {
 public:
  static EigenConcreteType<TPoint> impl(TPoint const& lhs, TPoint const& rhs) {
    return lhs.cwiseMax(rhs);
  }
};

template <class TScalar>
class Cast;

template <ScalarType TScalar>
class Cast<TScalar> {
 public:
  template <class TTo>
  static TTo impl(TScalar const& s) {
    return static_cast<TTo>(s);
  }
  template <class TTo>
  static TTo implScalar(TScalar const& s) {
    return static_cast<TTo>(s);
  }
};

template <EigenType TPoint>
class Cast<TPoint> {
 public:
  template <class TTo>
  static auto impl(TPoint const& v) {
    return v.template cast<typename TTo::Scalar>().eval();
  }
  template <class TTo>
  static auto implScalar(TPoint const& v) {
    return v.template cast<TTo>().eval();
  }
};

template <ScalarType TPoint>
class Cast<sophus::So3<TPoint>> {
 public:
  template <class TTo>
  static auto impl(sophus::So3<TPoint> const& v) {
    return v.template cast<typename TTo::Scalar>();
  }
  template <class TTo>
  static auto implScalar(sophus::So3<TPoint> const& v) {
    return v.template cast<TTo>();
  }
};

template <ScalarType TPoint>
class Cast<sophus::Se3<TPoint>> {
 public:
  template <class TTo>
  static auto impl(sophus::Se3<TPoint> const& v) {
    return v.template cast<typename TTo::Scalar>();
  }
  template <class TTo>
  static auto implScalar(sophus::Se3<TPoint> const& v) {
    return v.template cast<TTo>();
  }
};

template <ScalarType TPoint>
class Cast<std::vector<TPoint>> {
 public:
  template <class TTo>
  static auto impl(std::vector<TPoint> const& v) {
    using ToEl = std::decay_t<decltype(*std::declval<TTo>().data())>;
    std::vector<ToEl> r;
    r.reserve(v.size());
    for (auto const& el : v) {
      r.push_back(Cast<TPoint>::template impl<ToEl>(el));
    }
    return r;
  }
  template <class TTo>
  static auto implScalar(std::vector<TPoint> const& v) {
    using ToEl = decltype(Cast<TPoint>::template implScalar<TTo>(v[0]));
    std::vector<ToEl> r;
    r.reserve(v.size());
    for (auto const& el : v) {
      r.push_back(Cast<TPoint>::template impl<ToEl>(el));
    }
    return r;
  }
};

template <class TPoint>
class Zero;

template <ScalarType TScalar>
class Zero<TScalar> {
 public:
  static TScalar impl() { return static_cast<TScalar>(0); }
};

template <EigenType TPoint>
class Zero<TPoint> {
 public:
  static auto impl() { return TPoint::Zero().eval(); }
};

template <class TPoint>
class Eval;

template <ScalarType TScalar>
class Eval<TScalar> {
 public:
  static TScalar impl(TScalar const& s) { return s; }
};

template <EigenType TPoint>
class Eval<TPoint> {
 public:
  static auto impl(TPoint const& v) { return v.eval(); }
};

template <class TScalar>
class AllTrue;

template <ScalarType TScalar>
class AllTrue<TScalar> {
 public:
  static bool impl(TScalar const& s) { return bool(s); }
};

template <EigenDenseType TPoint>
class AllTrue<TPoint> {
 public:
  static bool impl(TPoint const& v) { return v.all(); }
};

template <class TScalar>
class AnyTrue;

template <ScalarType TScalar>
class AnyTrue<TScalar> {
 public:
  static bool impl(TScalar const& s) { return bool(s); }
};

template <EigenDenseType TPoint>
class AnyTrue<TPoint> {
 public:
  static bool impl(TPoint const& v) { return v.any(); }
};

template <class TScalar>
class IsFinite;

template <ScalarType TScalar>
class IsFinite<TScalar> {
 public:
  static bool impl(TScalar const& s) { return std::isfinite(s); }
};

template <EigenDenseType TPoint>
class IsFinite<TPoint> {
 public:
  static bool impl(TPoint const& v) { return v.isFinite().all(); }
};

template <class TScalar>
class IsNan;

template <ScalarType TScalar>
class IsNan<TScalar> {
 public:
  static bool impl(TScalar const& s) { return std::isnan(s); }
};

template <EigenDenseType TPoint>
class IsNan<TPoint> {
 public:
  static bool impl(TPoint const& p) { return p.array().isNaN().all(); }
};

template <class TScalar>
class Reduce;

template <ScalarType TScalar>
class Reduce<TScalar> {
 public:
  using Aggregate = TScalar;

  template <class TReduce, class TFunc>
  static void implUnary(TScalar const& s, TReduce& reduce, TFunc const& f) {
    f(s, reduce);
  }

  template <class TReduce, class TFunc>
  static void implBinary(
      TScalar const& a, TScalar const& b, TReduce& reduce, TFunc const& f) {
    f(a, b, reduce);
  }
};

template <EigenDenseType TPoint>
class Reduce<TPoint> {
 public:
  template <class TReduce, class TFunc>
  static void implUnary(TPoint const& v, TReduce& reduce, TFunc const& f) {
    for (int r = 0; r < v.rows(); ++r) {
      for (int c = 0; c < v.cols(); ++c) {
        f(v(r, c), reduce);
      }
    }
  }

  template <class TReduce, class TFunc>
  static void implBinary(
      TPoint const& a, TPoint const& b, TReduce& reduce, TFunc const& f) {
    for (int r = 0; r < a.rows(); ++r) {
      for (int c = 0; c < a.cols(); ++c) {
        f(a(r, c), b(r, c), reduce);
      }
    }
  }
};

}  // namespace details

template <PointType TPoint>
auto zero() {
  return details::Zero<TPoint>::impl();
}

template <class TTo, PointType TPoint>
auto cast(TPoint const& p) {
  return details::Cast<TPoint>::template impl<TTo>(p);
}

template <Arithmetic TTo, PointType TPoint>
auto cast(TPoint const& p) {
  return details::Cast<TPoint>::template implScalar<TTo>(p);
}

template <PointType TPoint>
auto eval(TPoint const& p) {
  return details::Eval<TPoint>::impl(p);
}

template <PointType TPoint>
bool allTrue(TPoint const& p) {
  return details::AllTrue<TPoint>::impl(p);
}

template <PointType TPoint>
bool anyTrue(TPoint const& p) {
  return details::AnyTrue<TPoint>::impl(p);
}

template <PointType TPoint>
bool isFinite(TPoint const& p) {
  return details::IsFinite<TPoint>::impl(p);
}

template <PointType TPoint>
bool isNan(TPoint const& p) {
  return details::IsNan<TPoint>::impl(p);
}

template <PointType TPoint>
auto square(TPoint const& v) {
  return details::Square<TPoint>::impl(v);
}

template <PointType TPoint>
TPoint min(TPoint const& a, TPoint const& b) {
  return details::Min<TPoint>::impl(a, b);
}

template <PointType TPoint>
TPoint max(TPoint const& a, TPoint const& b) {
  return details::Max<TPoint>::impl(a, b);
}

template <PointType TPoint>
TPoint clamp(TPoint const& val, TPoint const& a, TPoint const& b) {
  return sophus::max(a, sophus::min(val, b));
}

template <PointType TPoint, class TFunc, class TReduce>
void reduceArg(TPoint const& x, TReduce& reduce, TFunc&& func) {
  details::Reduce<TPoint>::impl_unary(x, reduce, std::forward<TFunc>(func));
}

template <PointType TPoint, class TFunc, class TReduce>
void reduceArg(
    TPoint const& a, TPoint const& b, TReduce& reduce, TFunc&& func) {
  details::Reduce<TPoint>::impl_binary(a, b, reduce, std::forward<TFunc>(func));
}

template <PointType TPoint, class TFunc, class TReduce>
TReduce reduce(TPoint const& x, TReduce const& initial, TFunc&& func) {
  TReduce reduce = initial;
  details::Reduce<TPoint>::impl_unary(x, reduce, std::forward<TFunc>(func));
  return reduce;
}

template <PointType TPoint, class TFunc, class TReduce>
TReduce reduce(
    TPoint const& a, TPoint const& b, TReduce const& initial, TFunc&& func) {
  TReduce reduce = initial;
  details::Reduce<TPoint>::impl_binary(a, b, reduce, std::forward<TFunc>(func));
  return reduce;
}

template <ScalarType TPoint>
auto floor(TPoint s) {
  using std::floor;
  return floor(s);
}

template <EigenDenseType TPoint>
auto floor(TPoint p) {
  for (auto& e : p.reshaped()) {
    e = sophus::floor(e);
  }
  return p;
}

template <ScalarType TPoint>
auto ceil(TPoint s) {
  using std::ceil;
  return ceil(s);
}

template <EigenDenseType TPoint>
auto ceil(TPoint p) {
  for (auto& e : p.reshaped()) {
    e = sophus::ceil(e);
  }
  return p;
}

template <ScalarType TPoint>
auto round(TPoint s) {
  using std::ceil;
  return ceil(s);
}

template <EigenDenseType TPoint>
auto round(TPoint p) {
  for (auto& e : p.reshaped()) {
    e = sophus::round(e);
  }
  return p;
}

template <ScalarType TPoint>
[[nodiscard]] auto plus(TPoint p, TPoint s) {
  p += s;
  return p;
}

template <EigenDenseType TPoint>
[[nodiscard]] auto plus(TPoint p, typename TPoint::Scalar s) {
  p.array() += s;
  return p;
}

template <ScalarType TPoint>
[[nodiscard]] bool isLessEqual(TPoint const& lhs, TPoint const& rhs) {
  return lhs <= rhs;
}

template <EigenDenseType TPoint>
[[nodiscard]] bool isLessEqual(TPoint const& lhs, TPoint const& rhs) {
  return allTrue(eval(lhs.array() <= rhs.array()));
}

template <ScalarType TPoint>
[[nodiscard]] Expected<TPoint> tryGetElem(
    TPoint const& p, size_t row, size_t col = 0) {
  if (row == 0 && col == 0) {
    return p;
  }
  return SOPHUS_UNEXPECTED("row ({}) and col ({}) must be 0", row, col);
}

template <EigenDenseType TPoint>
[[nodiscard]] Expected<TPoint> tryGetElem(
    TPoint const& p, size_t row, size_t col = 0) {
  if (row < p.rows() && col < p.cols()) {
    return p(row, col);
  }
  return SOPHUS_UNEXPECTED(
      "({}, {}) access of array of size {} x {}", row, col, p.rows(), p.cols());
}

template <ScalarType TPoint>
[[nodiscard]] Expected<Success> trySetElem(
    TPoint& p, TPoint s, size_t row, size_t col = 0) {
  if (row == 0 && col == 0) {
    p = s;
    return Success{};
  }
  return SOPHUS_UNEXPECTED("row ({}) and col ({}) must be 0", row, col);
}

template <EigenDenseType TPoint>
[[nodiscard]] Expected<Success> trySetElem(
    TPoint& p, typename TPoint::Scalar s, size_t row, size_t col = 0) {
  if (row == 0 && col == 0) {
    p(row, col) = s;
    return Success{};
  }
  return SOPHUS_UNEXPECTED("row ({}) and col ({}) must be 0", row, col);
}

}  // namespace sophus
