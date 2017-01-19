#!/usr/bin/env bash

# unofficial "bash strict mode"
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
# Commented out 'u' because the `activate` script has some unassignment issues
# set -u
set -eo pipefail
IFS=$'\n\t'

build_type="opt"

DIR=$(dirname $(realpath $0))

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

extra_opts=""
if [ $build_type == "opt" ]; then
    extra_opts="--copt -O3"
elif [ $build_type == "vtune" ]; then
    build_type="opt"
    extra_opts="--copt -g --copt -O3"
fi


echo "Building configuration $build_type"
#max_build_threads=$(bc <<< "scale=0; ($(nproc) * 1.1) / 1" )
max_build_threads=$(nproc)
eval "bazel build $extra_opts -j $max_build_threads -c $build_type //tensorflow/tools/pip_package:build_pip_package"
