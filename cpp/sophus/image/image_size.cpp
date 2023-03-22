// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/image/image_size.h"

namespace sophus {

auto ImageSize::contains(Eigen::Vector2i const& obs, int border) const -> bool {
  return obs.x() >= border && obs.x() < this->iwidth() - border &&
         obs.y() >= border && obs.y() < this->iheight() - border;
}

auto ImageSize::contains(Eigen::Vector2d const& obs, double border) const
    -> bool {
  return obs.x() >= -0.5 + border && obs.x() <= this->width - 0.5 - border &&
         obs.y() >= -0.5 + border && obs.y() <= this->height - 0.5 - border;
}

auto ImageSize::contains(Eigen::Vector2f const& obs, float border) const
    -> bool {
  return obs.x() >= -0.5f + border && obs.x() <= this->width - 0.5f - border &&
         obs.y() >= -0.5f + border && obs.y() <= this->height - 0.5f - border;
}
auto half(ImageSize image_size) -> ImageSize {
  return ImageSize((image_size.width + 1) / 2, (image_size.height + 1) / 2);
}

auto operator==(ImageSize const& lhs, ImageSize const& rhs) -> bool {
  return lhs.width == rhs.width && lhs.height == rhs.height;
}

auto operator!=(ImageSize const& lhs, ImageSize const& rhs) -> bool {
  return lhs.width != rhs.width || lhs.height != rhs.height;
}

auto operator<(ImageSize const& lhs, ImageSize const& rhs) -> bool {
  return std::make_pair(lhs.width, lhs.height) <
         std::make_pair(rhs.width, rhs.height);
}

auto operator<<(std::ostream& os, ImageSize const& image_size)
    -> std::ostream& {
  os << "[" << image_size.width << " x " << image_size.height << "]";
  return os;
}

}  // namespace sophus
