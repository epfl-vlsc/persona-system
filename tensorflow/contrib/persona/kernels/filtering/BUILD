
load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "filter",
    srcs = glob(["*.cc"] + ["*.h"] + ["Filtering*.cpp"]),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
            "//tensorflow/core:framework_headers_lib",
            "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
            "//tensorflow/contrib/persona/kernels/agd-format:agd_format",
            "//tensorflow/contrib/persona/kernels/lttng:trace",
            "//tensorflow/contrib/persona/kernels/snap-align:snap_lib"
            ],
    copts = ["-std=gnu++11", "-Wall", "-g", "-fPIC", "-static", "-Wl,-rpath='/usr/local/lib'"],
    linkopts = ["-L/usr/local/lib/", "-lantlr4-runtime" , "-Wl,-rpath='/usr/local/lib'"],
    includes = ["antlr/local/include/antlr4-runtime"],
    alwayslink=1,
)
