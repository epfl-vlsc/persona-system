# user ops
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)
#load("//tensorflow:tensorflow.bzl", "tf_copts")


tf_proto_library(
    name = "snap_protos",
    srcs = ["snap_proto.proto"],
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    py_api_version = 2,
)

cc_library(
    name = "snap_lib",
    srcs = glob(["snap/SNAPLib/*.cpp"]),
    includes = [".", "snap/SNAPLib"],
    #hdrs = ["snap/SNAPLib/GenomeIndex.h",
            #"snap/SNAPLib/Read.h",
            #"snap/SNAPLib/AlignmentResult.h",
            #"snap/SNAPLib/BaseAligner.h",
            #"snap/SNAPLib/AlignerOptions.h",
            #"snap/SNAPLib/Genome.h",
            #"snap/SNAPLib/SingleAligner.h",
            #"snap/SNAPLib/SAM.h",
            #"snap/SNAPLib/ProbabilityDistance.h",
            #],
    hdrs = glob(["snap/SNAPLib/*.h"]),
    linkopts = [
        "-lm", "-lrt", "-lz",
    ],
    visibility = ["//visibility:public"],
    copts = ["-pthread", "-MMD", "-msse", "-Wno-format"],
)

cc_binary(
    name = "snap_align.so",
    #includes = [".", "snap/SNAPLib"],
    srcs = glob(["*.cc"]),
    #copts = tf_copts(),
    linkopts = [
        "-Wl,-Bsymbolic",
        "-lm",
    ],
    linkstatic = 1,
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core:protos_all_cc",
            "//tensorflow/core:framework",
            ":snap_lib",],
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
