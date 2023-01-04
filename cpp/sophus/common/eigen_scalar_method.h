
// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "eigen_concepts.h"

#include <Eigen/Core>
#include <sophus/lie/se3.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace sophus {

namespace details {

// EigenDenseType may be a map or view or abstract base class or something.
// This is an alias for the corresponding concrete type with storage
template <EigenDenseType TT>
using EigenConcreteType = std::decay_t<decltype(std::declval<TT>().eval())>;

template <class TScalar>
class Square {
 public:
  static TScalar impl(TScalar const& v) { return v * v; }
};

template <EigenDenseType TT>
class Square<TT> {
 public:
  static auto impl(TT const& v) -> typename TT::Scalar {
    return v.squaredNorm();
  }
};

template <class TScalar>
class Min {
 public:
  static TScalar impl(TScalar const& lhs, TScalar const& rhs) {
    return std::min(lhs, rhs);
  }
};

template <EigenDenseType TT>
class Min<TT> {
 public:
  static EigenConcreteType<TT> impl(TT const& lhs, TT const& rhs) {
    return lhs.cwiseMin(rhs);
  }
};

template <class TScalar>
class Max {
 public:
  static TScalar impl(TScalar const& lhs, TScalar const& rhs) {
    return std::max(lhs, rhs);
  }
};

template <EigenDenseType TT>
class Max<TT> {
 public:
  static EigenConcreteType<TT> impl(TT const& lhs, TT const& rhs) {
    return lhs.cwiseMax(rhs);
  }
};

template <class TScalar>
class Cast {
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

template <EigenType TT>
class Cast<TT> {
 public:
  template <class TTo>
  static auto impl(TT const& v) {
    return v.template cast<typename TTo::Scalar>().eval();
  }
  template <class TTo>
  static auto implScalar(TT const& v) {
    return v.template cast<TTo>().eval();
  }
};

template <class TT>
class Cast<sophus::So3<TT>> {
 public:
  template <class TTo>
  static auto impl(sophus::So3<TT> const& v) {
    return v.template cast<typename TTo::Scalar>();
  }
  template <class TTo>
  static auto implScalar(sophus::So3<TT> const& v) {
    return v.template cast<TTo>();
  }
};

template <class TT>
class Cast<sophus::Se3<TT>> {
 public:
  template <class TTo>
  static auto impl(sophus::Se3<TT> const& v) {
    return v.template cast<typename TTo::Scalar>();
  }
  template <class TTo>
  static auto implScalar(sophus::Se3<TT> const& v) {
    return v.template cast<TTo>();
  }
};

template <class TT>
class Cast<std::vector<TT>> {
 public:
  template <class TTo>
  static auto impl(std::vector<TT> const& v) {
    using ToEl = std::decay_t<decltype(*std::declval<TTo>().data())>;
    std::vector<ToEl> r;
    r.reserve(v.size());
    for (auto const& el : v) {
      r.push_back(Cast<TT>::template impl<ToEl>(el));
    }
    return r;
  }
  template <class TTo>
  static auto implScalar(std::vector<TT> const& v) {
    using ToEl = decltype(Cast<TT>::template implScalar<TTo>(v[0]));
    std::vector<ToEl> r;
    r.reserve(v.size());
    for (auto const& el : v) {
      r.push_back(Cast<TT>::template impl<ToEl>(el));
    }
    return r;
  }
};

template <class TScalar>
class Zero {
 public:
  static TScalar impl() { return static_cast<TScalar>(0); }
};

template <EigenType TT>
class Zero<TT> {
 public:
  static auto impl() { return TT::Zero().eval(); }
};

template <class TScalar>
class Eval {
 public:
  static TScalar impl(TScalar const& s) { return s; }
};

template <EigenType TT>
class Eval<TT> {
 public:
  static auto impl(TT const& v) { return v.eval(); }
};

template <class TScalar>
class AllTrue {
 public:
  static bool impl(TScalar const& s) { return bool(s); }
};

template <EigenDenseType TT>
class AllTrue<TT> {
 public:
  static bool impl(TT const& v) { return v.all(); }
};

template <class TScalar>
class AnyTrue {
 public:
  static bool impl(TScalar const& s) { return bool(s); }
};

template <EigenDenseType TT>
class AnyTrue<TT> {
 public:
  static bool impl(TT const& v) { return v.any(); }
};

template <class TScalar>
class IsFinite {
 public:
  static bool impl(TScalar const& s) { return std::isfinite(s); }
};

template <EigenDenseType TT>
class IsFinite<TT> {
 public:
  static bool impl(TT const& v) { return v.isFinite().all(); }
};

template <class TScalar>
class Reduce {
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

template <EigenDenseType TT>
class Reduce<TT> {
 public:
  template <class TReduce, class TFunc>
  static void implUnary(TT const& v, TReduce& reduce, TFunc const& f) {
    for (int r = 0; r < v.rows(); ++r) {
      for (int c = 0; c < v.cols(); ++c) {
        f(v(r, c), reduce);
      }
    }
  }

  template <class TReduce, class TFunc>
  static void implBinary(
      TT const& a, TT const& b, TReduce& reduce, TFunc const& f) {
    for (int r = 0; r < a.rows(); ++r) {
      for (int c = 0; c < a.cols(); ++c) {
        f(a(r, c), b(r, c), reduce);
      }
    }
  }
};

}  // namespace details

template <class TT>
auto zero() {
  return details::Zero<TT>::impl();
}

template <class TTo, class TT>
auto cast(const TT& p) {
  return details::Cast<TT>::template impl<TTo>(p);
}

template <Arithmetic TTo, class TT>
auto cast(const TT& p) {
  return details::Cast<TT>::template implScalar<TTo>(p);
}

template <class TT>
auto eval(const TT& p) {
  return details::Eval<TT>::impl(p);
}

template <class TT>
bool allTrue(const TT& p) {
  return details::AllTrue<TT>::impl(p);
}

template <class TT>
bool anyTrue(const TT& p) {
  return details::AnyTrue<TT>::impl(p);
}

template <class TT>
bool isFinite(const TT& p) {
  return details::IsFinite<TT>::impl(p);
}

template <class TT>
auto square(const TT& v) {
  return details::Square<TT>::impl(v);
}

template <class TT>
TT min(const TT& a, const TT& b) {
  return details::Min<TT>::impl(a, b);
}

template <class TT>
TT max(const TT& a, const TT& b) {
  return details::Max<TT>::impl(a, b);
}

template <class TT, class TFunc, class TReduce>
void reduceArg(const TT& x, TReduce& reduce, TFunc&& func) {
  details::Reduce<TT>::impl_unary(x, reduce, std::forward<TFunc>(func));
}

template <class TT, class TFunc, class TReduce>
void reduceArg(const TT& a, const TT& b, TReduce& reduce, TFunc&& func) {
  details::Reduce<TT>::impl_binary(a, b, reduce, std::forward<TFunc>(func));
}

template <class TT, class TFunc, class TReduce>
TReduce reduce(const TT& x, TReduce const& initial, TFunc&& func) {
  TReduce reduce = initial;
  details::Reduce<TT>::impl_unary(x, reduce, std::forward<TFunc>(func));
  return reduce;
}

template <class TT, class TFunc, class TReduce>
TReduce reduce(const TT& a, const TT& b, TReduce const& initial, TFunc&& func) {
  TReduce reduce = initial;
  details::Reduce<TT>::impl_binary(a, b, reduce, std::forward<TFunc>(func));
  return reduce;
}

}  // namespace sophus
