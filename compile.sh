#!/usr/bin/env bash

max_build_threads=$(bc <<< "scale=0; ($(nproc) * 0.6) / 1" )
bazel build -j $max_build_threads -c opt --verbose_failures //tensorflow/tools/pip_package:build_pip_package

