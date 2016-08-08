# user ops

load("//tensorflow:tensorflow.bzl", "tf_copts")
# copy this
cc_library(
    name = "dense_format",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h", "json/src/json.hpp"]),
    copts = tf_copts() + ["-fexceptions"],
    nocopts = "-fno-exceptions",
    linkstatic = 1,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core:framework", "//tensorflow/core/kernels:io",
    "//tensorflow/core/user_ops/dna-align:snap_lib",
    "//tensorflow/core/user_ops/object-pool:object_pool",
    "//tensorflow/core/user_ops/lttng:trace",
    "//tensorflow/core/kernels:ops_util"],
    # TODO: this needs to be built better, like this: https://github.com/mzhaom/trunk/blob/master/third_party/boost/BUILD
    linkopts = [
             "-lz", "-ls3", "-lrados"
    ]
)
