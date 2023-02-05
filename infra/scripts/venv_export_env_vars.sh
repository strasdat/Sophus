#!/bin/bash

# Source this file (executing it wont set variables).
# Setup env vars for using libs and binaries in venv/prefix inside.
# Required for CMake and dynamic library loading on Linux.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
PREFIX_PATH="$ROOT_DIR/venv/prefix"

export PATH=$PREFIX_PATH/bin:$PATH
export CMAKE_PREFIX_PATH=$PREFIX_PATH
export LD_LIBRARY_PATH=$PREFIX_PATH/lib:${LD_LIBRARY_PATH}
