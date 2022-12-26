
// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/common/concept_utils.h"

#include <Eigen/Core>

namespace sophus {

// These concept let us match Eigen's CRTP pattern and capture the true Derived
// type safely

template <class TDerived>
concept EigenType = DerivedFrom<TDerived, Eigen::EigenBase<TDerived>>;

template <class TDerived>
concept EigenDenseType = DerivedFrom<TDerived, Eigen::DenseBase<TDerived>>;

template <class TDerived>
concept EigenMatrixType = DerivedFrom<TDerived, Eigen::MatrixBase<TDerived>>;

template <class TDerived>
concept EigenArrayType = DerivedFrom<TDerived, Eigen::ArrayBase<TDerived>>;

template <class TT1, typename TT2>
concept EigenSameDim = requires(TT1, TT2) {
  EigenDenseType<TT1>&& EigenDenseType<TT2> &&
      (TT1::RowsAtCompileTime == Eigen::Dynamic ||
       TT1::RowsAtCompileTime == TT2::RowsAtCompileTime) &&
      (TT1::ColsAtCompileTime == Eigen::Dynamic ||
       TT1::ColsAtCompileTime == TT2::ColsAtCompileTime);
};

template <int kRows, int kCols, typename TT>
concept EigenWithDim = requires(TT) {
  EigenDenseType<TT>&& TT::RowsAtCompileTime == kRows&& TT::ColsAtCompileTime ==
      kCols;
};

template <int kRows, int kCols, typename TT>
concept EigenWithDimOrDynamic = requires(TT) {
  EigenDenseType<TT> &&
      (TT::RowsAtCompileTime == Eigen::Dynamic ||
       TT::RowsAtCompileTime == kRows) &&
      (TT::ColsAtCompileTime == Eigen::Dynamic ||
       TT::ColsAtCompileTime == kCols);
};

}  // namespace sophus
