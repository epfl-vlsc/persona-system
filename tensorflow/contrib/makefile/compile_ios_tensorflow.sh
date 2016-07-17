#!/bin/bash -x
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# Builds the TensorFlow core library with ARM and x86 architectures for iOS, and
# packs them into a fat file.

GENDIR=tensorflow/contrib/makefile/gen/
LIBDIR=${GENDIR}lib
LIB_PREFIX=libtensorflow-core

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARMV7 LIB_NAME=${LIB_PREFIX}-armv7.a OPTFLAGS="$1" $2 $3
if [ $? -ne 0 ]
then
  echo "armv7 compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARMV7S LIB_NAME=${LIB_PREFIX}-armv7s.a OPTFLAGS="$1" $2 $3
if [ $? -ne 0 ]
then
  echo "arm7vs compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=ARM64 LIB_NAME=${LIB_PREFIX}-arm64.a OPTFLAGS="$1" $2 $3
if [ $? -ne 0 ]
then
  echo "arm64 compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=I386 LIB_NAME=${LIB_PREFIX}-i386.a OPTFLAGS="$1" $2 $3
if [ $? -ne 0 ]
then
  echo "i386 compilation failed."
  exit 1
fi

make -f tensorflow/contrib/makefile/Makefile cleantarget
make -f tensorflow/contrib/makefile/Makefile \
TARGET=IOS IOS_ARCH=X86_64 LIB_NAME=${LIB_PREFIX}-x86_64.a OPTFLAGS="$1" $2 $3
if [ $? -ne 0 ]
then
  echo "x86_64 compilation failed."
  exit 1
fi

lipo \
${LIBDIR}/${LIB_PREFIX}-armv7.a \
${LIBDIR}/${LIB_PREFIX}-armv7s.a \
${LIBDIR}/${LIB_PREFIX}-arm64.a \
${LIBDIR}/${LIB_PREFIX}-i386.a \
${LIBDIR}/${LIB_PREFIX}-x86_64.a \
-create \
-output ${LIBDIR}/${LIB_PREFIX}.a
