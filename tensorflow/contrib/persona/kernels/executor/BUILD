load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "executor",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/persona/kernels:lockfree",
    ],
    alwayslink = 1,
)
