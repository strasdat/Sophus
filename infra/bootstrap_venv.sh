#!/bin/bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $DIR/scripts/venv_activate.sh
$DIR/scripts/venv_update_prefix_deps.sh
