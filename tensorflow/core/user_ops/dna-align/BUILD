# user ops
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)
load("//tensorflow:tensorflow.bzl", "tf_copts")

tf_proto_library_cc(
    name = "snap_protos",
    srcs = ["snap_proto.proto"],
    cc_libs = ["//tensorflow/core:protos_all_cc"],
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    py_api_version = 2,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "snap_lib",
    srcs = glob(["snap/SNAPLib/*.cpp"]),
    hdrs = glob(["snap/SNAPLib/*.h"]),
    includes = ["snap/SNAPLib"],
    linkopts = [
        "-lm", "-lrt", "-lz"
    ],
    visibility = ["//visibility:public"],
    copts = ["-pthread", "-MMD", "-msse", "-Wno-format"]
)

cc_library(
    name = "snap_align_lib",
    hdrs = ["SnapAlignerWrapper.h"],
    includes = ["."],
    srcs = glob(["*.cc"]),
    copts = tf_copts(),
    #linkopts = [
        #"-Wl,-Bsymbolic",
        #"-lm"
    #],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
            "//tensorflow/core:framework",
            ":snap_protos_cc",
            ":snap_lib"],
    alwayslink=1,
)

#cc_binary(
#    name = "dna_align.so",
#    includes = ["snap/SNAPLib"],
#    srcs = glob(["*.cc"]),
#	#copts = ["-w"],
#    linkopts = [
#        "-Wl,-Bsymbolic",
#        "-lm",
#    ],
#    linkshared = 1,
#    linkstatic = 1,
#)
