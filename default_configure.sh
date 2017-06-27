#!/usr/bin/env bash

set -eu
set -o pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd "$DIR" >/dev/null

export PYTHON_BIN_PATH=$(which python3)
export CC_OPT_FLAGS="-march=native"
export TF_NEED_JEMALLOC=1
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_OPENCL=0
export TF_ENABLE_XLA=0
export TF_NEED_CUDA=0
export TF_NEED_MKL=0
export TF_NEED_VERBS=0
export PYTHON_LIB_PATH=/usr/lib/python3/dist-packages

# if you configure from within a docker container, and then outside it again
# there is this annoying file that gets owned by the file permissions in docker
# this will nuke it if that would otherwise cause configure to fail
root_gen_path="$DIR/tensorflow/tools/git/gen"
if [[ $EUID != 0 && -e "$root_gen_path" ]]; then
    file_owner=$(stat -c %u "$root_gen_path")
    if [[ $file_owner == 0 ]]; then
        sudo rm -rf "$root_gen_path"
    fi
fi

./configure

popd >/dev/null
