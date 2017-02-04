
load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "agdbam",
    srcs = glob(["*.cc"] + ["*.h"] ),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
            "//tensorflow/core:framework",
            "//tensorflow/core/user_ops/object-pool:object_pool",
            "//tensorflow/core/user_ops/agd-format:agd_format",
            "//tensorflow/core/user_ops/dna-align:snap_lib"
            ],
    #linkopts = ["-lz"],
    alwayslink=1,
)
