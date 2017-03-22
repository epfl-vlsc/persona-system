#!/usr/bin/env bash
set -eu
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -gt  1 ]; then
    echo "Illegal number of arguments: $#"
    exit 1
else
    if [ "$#" -eq 1 ]; then
        outdir="$1"
    else
        outdir="/tmp"
    fi
fi

if [ ! -d "$outdir" ]; then
    echo "pip output directory $outdir doesn't exist"
    exit 1
fi

PIP_PKG_BUILD_DIR=$(mktemp -d -p "$outdir")

trap '/bin/rm -rf "${PIP_PKG_BUILD_DIR}" >/dev/null 2>&1; exit 0' SIGTERM

pushd "$SCRIPT_DIR" >/dev/null
./default_configure.sh
./compile.sh
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package "$PIP_PKG_BUILD_DIR"
popd >/dev/null
echo "$PIP_PKG_BUILD_DIR"
exit 0
