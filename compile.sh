#!/usr/bin/env bash

# unofficial "bash strict mode"
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
# Commented out 'u' because the `activate` script has some unassignment issues
# set -u
set -eo pipefail
IFS=$'\n\t'

build_type="opt"

process_args() {
case "$#" in
    0)
        ;;
    1)
        build_type="$1"
        ;;
    *)
        echo "Script only accepts 0 or 1 argument!"
        exit 2
        ;;
esac
}

process_args "$@"

echo "Building configuration $build_type"
max_build_threads=$(bc <<< "scale=0; ($(nproc) * 0.6) / 1" )
bazel build -j $max_build_threads -c $build_type --verbose_failures //tensorflow/tools/pip_package:build_pip_package

