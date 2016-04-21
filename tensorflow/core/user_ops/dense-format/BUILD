# user ops

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)

load("//tensorflow:tensorflow.bzl", "tf_copts")
# copy this
cc_library(
    name = "dense_format",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts() + ["-fexceptions"],
    nocopts = "-fno-exceptions",
    linkstatic = 1,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core:framework", "//tensorflow/core/kernels:reader_base", "//tensorflow/core/user_ops/dna-align:snap_protos_cc", "//tensorflow/core/kernels:writer_base",
    "//tensorflow/core/kernels:ops_util"],
    # TODO: this needs to be built better, like this: https://github.com/mzhaom/trunk/blob/master/third_party/boost/BUILD
    linkopts = [
             "-lboost_iostreams"
    ]
)
