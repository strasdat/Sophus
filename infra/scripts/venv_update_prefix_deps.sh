#!/bin/bash
set -ex


# This will download the prebuilt dependencies for Sophus. It will first
# check if the binaries are already up-to-date based on the farm-ng-cmake git
# SHA. The binaries are placed in ../venv/prefix for the Sophus build,
# though you must set CMAKE_PREFIX_PATH to that directory for cmake to find them

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"

# Choose release SHA based on farm-ng-cmake git repo SHA
# Override by setting RELEASE_SHA env var
RELEASE_SHA="${RELEASE_SHA:-$(git submodule status ci-submodules/farm-ng-cmake | cut -c2-8)}"

# If the needed RELEASE_SHA doesn't match what we have, download
if ! grep -q $RELEASE_SHA "$ROOT_DIR/venv/prefix/release_version.txt"; then
  mkdir -p "$ROOT_DIR/venv"
  $ROOT_DIR/ci-submodules/farm-ng-cmake/scripts/download_release.sh
  rm -rf $ROOT_DIR/venv/prefix
  tar -xzf $ROOT_DIR/ci-submodules/farm-ng-cmake/scripts/venv.tar.gz --strip-components=1 -C $ROOT_DIR/venv

  if [[ "$OSTYPE" == "darwin"* ]]; then
    # MacOS RPath works a bit differently. We need to define the RPaths that
    # will be searched at runtime. We'll add a relative one to each executable
    for exe in "$ROOT_DIR/venv/prefix/bin"/*; do
      install_name_tool -add_rpath @executable_path/../lib $exe || true
    done
  fi

  echo "$RELEASE_SHA" > $ROOT_DIR/venv/prefix/release_version.txt
  echo "[Updated binary dependencies. Done!]"
else
  echo "[Updated binary dependencies. Already up to date!]"
fi
