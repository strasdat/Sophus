// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#pragma once

#include "sophus/image/image.h"
#include "sophus/image/image_view.h"

namespace sophus {

// Bilinearly interpolates ``img`` at ``uv``.
//
// Preconditions:
//  - uv must be inside the image boundary
template <class TT>
auto interpolate(sophus::ImageView<TT> const& image, Eigen::Vector2f uv) -> TT {
  sophus::ImageSize image_size = image.imageSize();
  SOPHUS_ASSERT(
      image_size.contains(uv), "{} vs {}", image_size, uv.transpose());
  float iu = NAN;
  float iv = NAN;
  float frac_u = std::modf(uv.x(), &iu);
  float frac_v = std::modf(uv.y(), &iv);
  int u = iu;
  int v = iv;

  bool u_corner_case = u == image_size.width - 1;
  bool v_corner_case = v == image_size.height - 1;

  TT val00 = image(u, v);
  TT val01 = v_corner_case ? val00 : image(u, v + 1);
  TT val10 = u_corner_case ? val00 : image(u + 1, v);
  TT val11 = u_corner_case || v_corner_case ? val00 : image(u + 1, v + 1);

  TT val = (1.f - frac_u) * (1.f - frac_v) * val00  //
           + (1.f - frac_u) * frac_v * val01        //
           + frac_u * (1.f - frac_v) * val10        //
           + frac_u * frac_v * val11;
  return val;
}

}  // namespace sophus
