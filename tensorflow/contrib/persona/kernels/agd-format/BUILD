# user ops

load("//tensorflow:tensorflow.bzl", "tf_copts")

# copy this
cc_library(
    name = "agd_format",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts() + ["-fexceptions"],
    linkopts = [
        "-lz",
        "-lrados",
        "-lboost_system",
    ],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        #"//tensorflow/core/kernels:io",
        "//tensorflow/contrib/persona/kernels/snap-align:snap_lib",
        "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
        "//tensorflow/contrib/persona/kernels/lttng:trace",
        "//third_party/eigen3",
        "@protobuf//:protobuf",
        #"//tensorflow/core/kernels:ops_util"
    ],
    alwayslink = 1,
)
