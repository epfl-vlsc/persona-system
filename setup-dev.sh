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
max_build_threads=$(nproc)

prep_virtualenv() {
    install_dir=$1
    python_path=$2

    virtualenv --system-site-packages --python=$python_path $install_dir
    source $install_dir/bin/activate
}

build_tensorflow() {
    echo "Building using $max_build_threads threads"
    git submodule update --init # in case you forget, or switched branches
    #PYTHON_BIN_PATH=$(which python3) TF_NEED_GCP=0 TF_NEED_CUDA=0 ./configure
    bazel build -j $max_build_threads //tensorflow/tools/pip_package:build_pip_package
}

prep_dirs() {
    if [[ $OSTYPE == darwin* ]]; then
        echo "Just....don't install on Mac. You're gonna have a bad time :/"
        exit 2
        sudo easy_install pip
    else
        # assume Ubuntu
        # have to do this in case of ceph-common complaining
        set +e
        sudo apt-get -y install python3-pip python3-dev python-virtualenv
        set -e
        sudo pip3 install --upgrade pip setuptools wheel pysam
    fi
    [ ! -e $build_virtualenv_dir ] || rm -rf $build_virtualenv_dir
    [ ! -e $dev_dir ] || rm -rf $dev_dir
    mkdir $build_virtualenv_dir
    mkdir $dev_dir
}

install_dev_build() {
    pushd .
    cd $build_virtualenv_dir

    ln -sf ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/* .
    ln -sf ../tensorflow/tools/pip_package/* .
    python3 setup.py develop

    popd
    deactivate
}

build_tensorflow
prep_dirs
prep_virtualenv $dev_dir $(which python3)
install_dev_build
