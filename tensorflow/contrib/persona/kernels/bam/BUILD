
load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "agdbam",
    srcs = glob(["*.cc"] + ["*.h"] ),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
            "//tensorflow/core:framework_headers_lib",
            "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
            "//tensorflow/contrib/persona/kernels/agd-format:agd_format",
            "//tensorflow/contrib/persona/kernels/concurrent_queue:concurrent_queue",
            "//tensorflow/contrib/persona/kernels/snap-align:snap_lib"
            ],
    #linkopts = ["-lz"],
    alwayslink=1,
)
