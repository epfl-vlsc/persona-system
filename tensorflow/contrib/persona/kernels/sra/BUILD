
load("//tensorflow:tensorflow.bzl", "tf_copts")

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
    linkopts = ["-L/usr/local/ncbi/ncbi-vdb/lib64/", "-L/usr/local/ngs/ngs-sdk/lib64","-lncbi-ngs-c++", "-lngs-c++", "-lncbi-ngs-c++", "-lncbi-vdb-static", "-lngs-adapt-c++", "-lpthread", "-ldl", "-lm" ],
    alwayslink=1,
)
