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
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
