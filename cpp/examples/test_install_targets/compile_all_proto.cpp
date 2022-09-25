
// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/geometry/proto/conv.h"
#include "sophus/image/proto/conv.h"
#include "sophus/lie/proto/conv.h"
#include "sophus/linalg/proto/conv.h"
#include "sophus/sensor/proto/conv.h"

#include <farm_ng/core/misc/proto/conv.h>
#include <farm_ng/core/prototools/proto_reader_writer.h>

using namespace farm_ng;

int main() {
  Uri uri;
  uri.path = "foo";

  auto proto = toProto(uri);

  writeProtobufToJsonFile("/tmp/foo.json", proto);
}
