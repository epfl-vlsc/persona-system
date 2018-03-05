# sorting operations

load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "protein_clustering",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    #linkopts = ["-lrados"],
    #linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/contrib/persona/kernels/protein-cluster/swps3:swps3_lib",
            "//tensorflow/core:framework_headers_lib",
            "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
            "//tensorflow/contrib/persona/kernels/agd-format:agd_format",
            ]
)
