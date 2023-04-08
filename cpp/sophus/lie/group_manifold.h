// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/lie/interp/average.h"
#include "sophus/lie/lie_group.h"

namespace sophus {
SOPHUS_ENUM(ManifoldPlusType, (left_plus, right_plus));

template <concepts::LieGroup TGroup, ManifoldPlusType kPlus>
struct GroupManifold {
  using Group = TGroup;
  using Tangent = typename TGroup::Tangent;
  using Params = typename Group::Params;
  using Scalar = typename TGroup::Scalar;
  static int constexpr kDof = TGroup::kDof;
  static int constexpr kNumParams = TGroup::kNumParams;

  // Constructor

  GroupManifold() = default;
  GroupManifold(GroupManifold const& g) = default;
  GroupManifold& operator=(GroupManifold const& g) = default;

  GroupManifold(Group const& g) : group(g) {}

  static auto fromParams(Params const& params) -> GroupManifold {
    return GroupManifold(Group::fromParams(params));
  }

  auto setParams(Params const& params) -> void { group.setParams(params); }

  auto unsafeMutPtr() -> Scalar* { return group.unsafeMutPtr(); }

  auto ptr() -> Scalar const* { return group.ptr(); }

  // oplus (either "left translation" or "right translation")
  auto oplus(Tangent const& a) const -> GroupManifold {
    if constexpr (kPlus == ManifoldPlusType::left_plus) {
      return group.leftPlus(a);
    } else {
      return group.rightPlus(a);
    }
  }

  // ominus (inverse operation of oplus, corresponding to the logarithm map)
  auto ominus(GroupManifold const& h) const -> Tangent {
    if constexpr (kPlus == ManifoldPlusType::left_plus) {
      return group.leftMinus(h.group);
    } else {
      return group.rightMinus(h.group);
    }
  }

  static auto tangentExamples() -> std::vector<Tangent> {
    return Group::tangentExamples();
  }

  auto params() const -> Params const& { return group.params(); }

  template <concepts::Range TSequenceContainer>
  static auto average(TSequenceContainer const& range)
      -> std::optional<GroupManifold> {
    std::vector<Group> groups;
    for (auto const& m : range) {
      groups.push_back(m.group);
    }

    auto maybe_mean = sophus::average(groups);
    if (maybe_mean) {
      return GroupManifold(*maybe_mean);
    }
    return std::nullopt;
  }

  Group group;
};

template <concepts::LieGroup TGroup>
using LeftPlus = GroupManifold<TGroup, ManifoldPlusType::left_plus>;

template <concepts::LieGroup TGroup>
using RightPlus = GroupManifold<TGroup, ManifoldPlusType::right_plus>;

}  // namespace sophus
