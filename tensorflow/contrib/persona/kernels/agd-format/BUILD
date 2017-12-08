# user ops

load("//tensorflow:tensorflow.bzl", "tf_copts")

# copy this
cc_library(
    name = "agd_format",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    linkopts = [
        "-lz",
        "-lrados",
        "-lboost_system",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        #"//tensorflow/core/kernels:io",
        "//tensorflow/contrib/persona/kernels/snap-align:snap_lib",
        "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
        "//tensorflow/contrib/persona/kernels/lttng:trace",
        "//tensorflow/contrib/persona/kernels/agd-format/proto:alignment_proto_cc",
        "//third_party/eigen3",
        #"@protobuf//:protobuf",
        "@protobuf_archive//:protobuf",
        #"//tensorflow/core/kernels:ops_util"
    ],
    alwayslink = 1,
)
