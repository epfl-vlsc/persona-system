load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "barrier",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        #"@protobuf//:protobuf",
        "@protobuf_archive//:protobuf",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
)
