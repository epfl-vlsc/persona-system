#!/usr/bin/env bash

# unofficial "bash strict mode"
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
# Commented out 'u' because the `activate` script has some unassignment issues
# set -u
set -eo pipefail
IFS=$'\n\t'

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

build_virtualenv_dir="${DIR}/_python_build"
dev_dir="${DIR}/python_dev"
max_build_threads=$(bc <<< "scale=0; ($(nproc) * 1.1) / 1" )

prep_virtualenv() {
    if [[ $OSTYPE == darwin* ]]; then
        sudo easy_install pip
    else
        # assume Ubuntu
        sudo apt-get install python-pip python-dev python-retrying python-ipaddr bc python-virtualenv
    fi

    virtualenv --system-site-packages --python=$(which python2) $dev_dir
    source $dev_dir/bin/activate
}

build_tensorflow() {
    echo "Building using $max_build_threads threads"
    git submodule update --init # in case you forget, or switched branches
    PYTHON_BIN_PATH=$(which python) TF_NEED_GCP=0 TF_NEED_CUDA=0 ./configure
    bazel build -j $max_build_threads //tensorflow/tools/pip_package:build_pip_package
}

prep_dirs() {
    [ -e $build_virtualenv_dir ] || mkdir $build_virtualenv_dir
    [ -e $dev_dir ] || mkdir $dev_dir
}

install_dev_build() {
    pushd .
    cd $build_virtualenv_dir
    
    ln -sf ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/* .
    ln -sf ../tensorflow/tools/pip_package/* .
    python setup.py develop

    popd
}

build_tensorflow
prep_dirs
prep_virtualenv
install_dev_build
deactivate
