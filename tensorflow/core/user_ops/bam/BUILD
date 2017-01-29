
load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "bamtoolsapi",
    #srcs = glob(["libsnap.a"]),
    srcs = glob(["bamtools/src/api/**/*.cpp"], exclude=["bamtools/src/api/internal/io/TcpSocketEngine_win_p.cpp"]),
    hdrs = glob(["bamtools/src/api/**/*.h", "bamtools/src/shared/**/*.h" ]),
    includes = ["tensorflow/core/user_ops/bam/bamtools/src/**/*"],
    copts = ["-Itensorflow/core/user_ops/bam/bamtools/src/"],
    linkopts = [
        "-lz"
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    #copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"]
)

cc_library(
    name = "agdbam",
    srcs = glob(["*.cc"]),
    includes = ["tensorflow/core/user_ops/bam/bamtools/src/**/*"],
    copts = ["-Itensorflow/core/user_ops/bam/bamtools/src/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
            "//tensorflow/core:framework",
            ":bamtoolsapi",
            "//tensorflow/core/user_ops/object-pool:object_pool",
            "//tensorflow/core/user_ops/agd-format:agd_format"
            ],
    alwayslink=1,
)
