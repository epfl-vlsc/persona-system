
load("//tensorflow:tensorflow.bzl", "tf_copts")

# this requires package ncbi-vdb which is not available on 16.04 Ubuntu. 
cc_library(
    name = "agdsra",
    srcs = glob(["*.cc"] + ["*.h"] ),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
            "//tensorflow/core:framework_headers_lib",
            "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
            "//tensorflow/contrib/persona/kernels/agd-format:agd_format",
            ],
    linkopts = ["-lngs-c++", "-lncbi-vdb-static", "-lngs-adapt-c++", "-lpthread", "-ldl", "-lm" ],
    alwayslink=1,
)
