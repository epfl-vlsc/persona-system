#!/usr/bin/env bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Build the Python PIP installation package for TensorFlow and install
# the package.
# The PIP installation is done using the --user flag.
#
# Usage:
#   pip.sh CONTAINER_TYPE [--test_tutorials]
#
# When executing the Python unit tests, the script obeys the shell
# variables: TF_BUILD_BAZEL_CLEAN, NO_TEST_ON_INSTALL
#
# TF_BUILD_BAZEL_CLEAN, if set to any non-empty and non-0 value, directs the
# script to perform bazel clean prior to main build and test steps.
#
# If NO_TEST_ON_INSTALL has any non-empty and non-0 value, the test-on-install
# part will be skipped.
#
# I the --test_tutorials flag is set, it will cause the script to run the
# tutorial tests (see test_tutorials.sh) after the PIP
# installation and the Python unit tests-on-install step.
#

# Helper functions
# Get the absolute path from a path
abs_path() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}


# Exit after a failure
die() {
    echo $@
    exit 1
}


# Get the command line arguments
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )

if [[ ! -z "${TF_BUILD_BAZEL_CLEAN}" ]] && \
   [[ "${TF_BUILD_BAZEL_CLEAN}" != "0" ]]; then
  echo "TF_BUILD_BAZEL_CLEAN=${TF_BUILD_BAZEL_CLEAN}: Performing 'bazel clean'"
  bazel clean
fi

DO_TEST_TUTORIALS=0
for ARG in $@; do
  if [[ "${ARG}" == "--test_tutorials" ]]; then
    DO_TEST_TUTORIALS=1
  fi
done

PIP_BUILD_TARGET="//tensorflow/tools/pip_package:build_pip_package"
if [[ ${CONTAINER_TYPE} == "cpu" ]]; then
  bazel build -c opt ${PIP_BUILD_TARGET} || die "Build failed."
elif [[ ${CONTAINER_TYPE} == "gpu" ]]; then
  bazel build -c opt --config=cuda ${PIP_BUILD_TARGET} || die "Build failed."
else
  die "Unrecognized container type: \"${CONTAINER_TYPE}\""
fi

echo "PY_TEST_WHITELIST: ${PY_TEST_WHITELIST}"
echo "PY_TEST_BLACKLIST: ${PY_TEST_BLACKLIST}"
echo "PY_TEST_GPU_BLACKLIST: ${PY_TEST_GPU_BLACKLIST}"

# Append GPU-only test blacklist
if [[ ${CONTAINER_TYPE} == "gpu" ]]; then
  PY_TEST_BLACKLIST="${PY_TEST_BLACKLIST}:${PY_TEST_GPU_BLACKLIST}"
fi

# If still in a virtualenv, deactivate it first
if [[ ! -z "$(which deactivate)" ]]; then
  echo "It appears that we are already in a virtualenv. Deactivating..."
  deactivate || die "FAILED: Unable to deactivate from existing virtualenv"
fi

# Obtain the path to Python binary
source tools/python_bin_path.sh

# Assume: PYTHON_BIN_PATH is exported by the script above
if [[ -z "$PYTHON_BIN_PATH" ]]; then
  die "PYTHON_BIN_PATH was not provided. Did you run configure?"
fi

# Determine the major and minor versions of Python being used (e.g., 2.7)
# This info will be useful for determining the directory of the local pip
# installation of Python
PY_MAJOR_MINOR_VER=$(${PYTHON_BIN_PATH} -V 2>&1 | awk '{print $NF}' | cut -d. -f-2)

echo "Python binary path to be used in PIP install: ${PYTHON_BIN_PATH} "\
"(Major.Minor version: ${PY_MAJOR_MINOR_VER})"

# Build PIP Wheel file
PIP_TEST_ROOT="pip_test"
PIP_WHL_DIR="${PIP_TEST_ROOT}/whl"
PIP_WHL_DIR=$(abs_path ${PIP_WHL_DIR})  # Get absolute path
rm -rf ${PIP_WHL_DIR} && mkdir -p ${PIP_WHL_DIR}
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PIP_WHL_DIR} || \
die "build_pip_package FAILED"

# Perform installation
WHL_PATH=$(ls ${PIP_WHL_DIR}/tensorflow*.whl)
if [[ $(echo ${WHL_PATH} | wc -w) -ne 1 ]]; then
  die "ERROR: Failed to find exactly one built TensorFlow .whl file in "\
"directory: ${PIP_WHL_DIR}"
fi

echo "whl file path = ${WHL_PATH}"

# Install, in user's local home folder
echo "Installing pip whl file: ${WHL_PATH}"

# Create temporary directory for install test
VENV_DIR="${PIP_TEST_ROOT}/venv"
rm -rf "${VENV_DIR}" && mkdir -p "${VENV_DIR}"
echo "Create directory for virtualenv: ${VENV_DIR}"

# Verify that virtualenv exists
if [[ -z $(which virtualenv) ]]; then
  die "FAILED: virtualenv not available on path"
fi

virtualenv -p "${PYTHON_BIN_PATH}" "${VENV_DIR}" ||
die "FAILED: Unable to create virtualenv"

source "${VENV_DIR}/bin/activate" ||
die "FAILED: Unable to activate virtualenv"

# Install the pip file in virtual env
pip install -v ${WHL_PATH} \
&& echo "Successfully installed pip package ${WHL_PATH}" \
|| die "pip install (without --upgrade) FAILED"

# If NO_TEST_ON_INSTALL is set to any non-empty value, skip all Python
# tests-on-install and exit right away
if [[ ! -z "${NO_TEST_ON_INSTALL}" ]] &&
   [[ "${NO_TEST_ON_INSTALL}" != "0" ]]; then
  echo "NO_TEST_ON_INSTALL=${NO_TEST_ON_INSTALL}:"
  echo "  Skipping ALL Python unit tests on install"
  exit 0
fi

# Call test_installation.sh to perform test-on-install
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${DIR}/test_installation.sh" --virtualenv ||
die "PIP tests-on-install FAILED"

# Optional: Run the tutorial tests
if [[ "${DO_TEST_TUTORIALS}" == "1" ]]; then
  "${DIR}/test_tutorials.sh" --virtualenv ||
die "PIP tutorial tests-on-install FAILED"
fi

deactivate ||
die "FAILED: Unable to deactivate virtualenv"
