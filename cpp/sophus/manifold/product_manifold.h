// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/concepts/manifold.h"

namespace sophus {

// Credit: @bogdan at http://stackoverflow.com/q/37373602/6367128
template <int... Ds>
constexpr std::array<int, sizeof...(Ds)> cumulativeSum() {
  int v = 0;
  return {{v += Ds...}};
}

template <concepts::Manifold... TSubManifold>
class ProductManifold {
 public:
  using Self = ProductManifold<TSubManifold...>;

  using Tuple = std::tuple<TSubManifold...>;
  using Scalars = std::tuple<typename TSubManifold::Scalar...>;

  static constexpr size_t kNumManifolds = sizeof...(TSubManifold);
  static constexpr std::array<int, kNumManifolds> kManifoldSizes = {
      {TSubManifold::kDof...}};
  static constexpr std::array<int, kNumManifolds + 1> kManifoldStarts =
      cumulativeSum<0, TSubManifold::kDof...>();

  using Scalar = typename std::tuple_element<0, Scalars>::type;

  static constexpr size_t kNumParams = [](auto const&... sizes) {
    size_t sum = 0;
    (..., (sum += sizes));
    return sum;
  }(TSubManifold::kNumParams...);

  static constexpr size_t kDof = [](auto const&... sizes) {
    size_t sum = 0;
    (..., (sum += sizes));
    return sum;
  }(TSubManifold::kDof...);

  // Return ith diagonal block of matrix representing covariance or similar.
  template <size_t i, class Derived>
  static auto subBlock(Eigen::MatrixBase<Derived> const& mat) {
    return mat
        .template block<kManifoldSizes[i], kManifoldSizes[i]>(
            kManifoldStarts[i], kManifoldStarts[i])
        .eval();
  }

  using Tangent = Eigen::Vector<Scalar, kDof>;

  ProductManifold() = default;
  ProductManifold(ProductManifold const&) = default;
  ProductManifold& operator=(ProductManifold const&) = default;

  ProductManifold(TSubManifold const&... manifolds)
      : manifolds_(manifolds...) {}

  // Return ith sub-manifold
  template <size_t i>
  auto& subManifold() {
    return std::get<i>(manifolds_);
  }

  // Return ith sub-manifold
  template <size_t i>
  auto const& subManifold() const {
    return std::get<i>(manifolds_);
  }

  auto oplus(Tangent const& tangent) const -> Self {
    Self result = *this;
    oplusImpl(
        [](auto const& manifold, auto const& tangent) {
          return manifold.oplus(tangent);
        },
        result.manifolds_,
        tangent);
    return result;
  }

  auto ominus(Self const& other) const -> Tangent {
    Tangent tangent;

    ominusImpl(
        [](auto const& manifold, auto const& other) {
          return manifold.ominus(other);
        },
        this->manifolds_,
        other.manifolds_,
        tangent);
    return tangent;
  }

  static auto tangentExamples() -> std::vector<Tangent> {
    std::vector<Tangent> out;
    Tangent t;
    getTangent(t);
    out.push_back(t);
    return out;
  }

  template <size_t kArrayLen>
  static std::optional<ProductManifold<TSubManifold...>> average(
      std::array<ProductManifold<TSubManifold...>, kArrayLen> const& range) {
    SOPHUS_ASSERT_GE(kArrayLen, 0);
    ProductManifold<TSubManifold...> result;
    averageImpl<0, kArrayLen>(result, range);
    return result;
  }

 private:
  template <int k, size_t kArrayLen>
  static auto averageImpl(
      ProductManifold<TSubManifold...>& result,
      std::array<ProductManifold<TSubManifold...>, kArrayLen> const& range);

  template <size_t i = 0>
  static auto getTangent(Tangent& t) {
    auto v = std::tuple_element_t<i, Tuple>::tangentExamples();
    getBlock<i>(t) = v[0];
    if constexpr (i < sizeof...(TSubManifold) - 1) {
      // recurse if not at end
      getTangent<i + 1>(t);
    }
    return t;
  }

  template <size_t i = 0>
  void oplusImpl(auto const& func, Tuple& out, Tangent const& in) const {
    getBlock<i>(out) = func(getBlock<i>(manifolds_), getBlock<i>(in));

    if constexpr (i < sizeof...(TSubManifold) - 1) {
      oplusImpl<i + 1>(func, out, in);
    }
  }

  template <size_t i = 0>
  void ominusImpl(
      auto const& func,
      Tuple const& out,
      Tuple const& in,
      Tangent& tangent) const {
    getBlock<i>(tangent) = func(getBlock<i>(out), getBlock<i>(in));

    if constexpr (i < sizeof...(TSubManifold) - 1) {
      ominusImpl<i + 1>(func, out, in, tangent);
    }
  }

  template <size_t i>
  static auto& getBlock(Tuple& g) {
    return std::get<i>(g);
  }

  template <size_t i>
  static auto const& getBlock(Tuple const& g) {
    return std::get<i>(g);
  }

  template <size_t i>
  static auto getBlock(Tangent& tangent) {
    constexpr size_t offset = kManifoldStarts[i];
    constexpr size_t size = kManifoldSizes[i];
    return tangent.template segment<size>(offset);
  }

  template <size_t i>
  static auto getBlock(Tangent const& tangent) {
    constexpr size_t offset = kManifoldStarts[i];
    constexpr size_t size = kManifoldSizes[i];
    return tangent.template segment<size>(offset);
  }

  Tuple manifolds_;
};

namespace details {
template <int k, size_t kArrayLen, class... TManifold>
struct ContainerAdapter {
  using Container = ProductManifold<TManifold...>;
  using ContainerArray = std::array<Container, kArrayLen>;
  using value_type = std::decay_t<
      decltype(std::declval<Container>().template subManifold<k>())>;

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type =
        typename ContainerAdapter<k, kArrayLen, TManifold...>::value_type;
    using pointer = value_type const*;
    using reference = value_type const&;

    Iterator(typename ContainerArray::const_iterator it) : it_(it) {}

    reference operator*() const { return it_->template subManifold<k>(); }
    pointer operator->() { return &operator*(); }

    // Prefix increment
    Iterator& operator++() {
      it_++;
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(Iterator const& a, Iterator const& b) {
      return a.it_ == b.it_;
    };
    friend bool operator!=(Iterator const& a, Iterator const& b) {
      return a.it_ != b.it_;
    };

   private:
    typename ContainerArray::const_iterator it_;
  };

  ContainerAdapter(std::array<Container, kArrayLen> const& ref) : ref(ref) {}

  using const_iterator = Iterator;
  using iterator = Iterator;
  Iterator cbegin() const { return Iterator(ref.cbegin()); }
  Iterator cend() const { return Iterator(ref.cend()); }
  Iterator begin() const { return cbegin(); }
  Iterator end() const { return cend(); }

  std::array<Container, kArrayLen> const& ref;
};
}  // namespace details

template <concepts::Manifold... TSubManifold>
template <int k, size_t kArrayLen>
auto ProductManifold<TSubManifold...>::averageImpl(
    ProductManifold<TSubManifold...>& result,
    std::array<ProductManifold<TSubManifold...>, kArrayLen> const& range) {
  details::ContainerAdapter<k, kArrayLen, TSubManifold...> adapter(range);
  auto const maybe_avg = std::tuple_element_t<
      k,
      typename ProductManifold<TSubManifold...>::Tuple>::average(adapter);
  result.template subManifold<k>() = FARM_UNWRAP(maybe_avg);

  if constexpr (k < sizeof...(TSubManifold) - 1) {
    // recurse if not at end
    averageImpl<k + 1, kArrayLen>(result, range);
  }
  return std::nullopt;
}

}  // namespace sophus
