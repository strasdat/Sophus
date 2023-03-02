// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Interpolation for Lie groups.

#pragma once

#include "sophus/lie/interp/details/interpolate_impl.h"

#include <Eigen/Eigenvalues>

namespace sophus {

/// This function interpolates between two Lie group elements
/// ``foo_from_bar`` and ``foo_from_daz`` with an interpolation factor
/// of ``alpha`` in [0, 1].
///
/// It returns a pose ``foo_T_quiz`` with ``quiz`` being a frame between ``bar``
/// and ``baz``. If ``alpha=0`` it returns ``foo_from_bar``. If it is 1, it
/// returns ``foo_from_daz``.
///
/// (Since interpolation on Lie groups is inverse-invariant, we can equivalently
/// think of the input arguments as being ``bar_from_foo``,
/// ``baz_from_foo`` and the return value being ``quiz_T_foo``.)
///
/// Precondition: ``p`` must be in [0, 1].
///
template <class TGroup, class TScalar2 = typename TGroup::Scalar>
auto interpolate(
    TGroup const& foo_from_bar,
    TGroup const& foo_from_daz,
    TScalar2 p = TScalar2(0.5f))
    -> std::enable_if_t<interp_details::Traits<TGroup>::kSupported, TGroup> {
  using Scalar = typename TGroup::Scalar;
  Scalar inter_p(p);
  SOPHUS_ASSERT(
      inter_p >= Scalar(0) && inter_p <= Scalar(1),
      "p ({}) must in [0, 1].",
      inter_p);
  return foo_from_bar *
         TGroup::exp(inter_p * (foo_from_bar.inverse() * foo_from_daz).log());
}

}  // namespace sophus
