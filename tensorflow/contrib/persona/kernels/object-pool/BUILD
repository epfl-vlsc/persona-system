load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "object_pool",
    srcs = glob(["*.cc"] + ["*.h"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts() + ["-fexceptions"],
    nocopts = "-fno-exceptions",
    linkstatic = 1,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core:framework_headers_lib", 
            #"//tensorflow/core/kernels:io",
            #"//tensorflow/core/kernels:ops_util"
            ],
)
