#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

build_virtualenv_dir="${DIR}/_python_build"
dev_dir="${DIR}/python_dev"
max_build_threads=$(bc <<< "scale=0; ($(nproc) * 0.6) / 1" )

prep_virtualenv() {
    if [[ $OSTYPE == darwin* ]]; then
        sudo easy_install pip
    else
        # assume Ubuntu
        sudo apt-get install python-pip python-dev
    fi
    sudo pip install --upgrade virtualenv

    virtualenv --system-site-packages $dev_dir
    source $dev_dir/bin/activate
}

build_tensorflow() {
    echo "Building using $max_build_threads threads"
    bazel build -j $max_build_threads -c opt //tensorflow/tools/pip_package:build_pip_package
}

prep_dirs() {
    [ -e $build_virtualenv_dir ] || mkdir $build_virtualenv_dir
    [ -e $dev_dir ] || mkdir $dev_dir
}

install_dev_build() {
    pushd .
    cd $build_virtualenv_dir

    ln -s ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/* .
    ln -s ../tensorflow/tools/pip_package/* .
    python setup.py develop

    popd
}

build_tensorflow
prep_dirs
prep_virtualenv
install_dev_build
deactivate
