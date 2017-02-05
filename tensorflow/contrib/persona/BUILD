# Description:
#   Contains ops to build an input pipeline for tensorflow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_cc_tests",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)

tf_custom_op_library(
    name = "python/ops/_persona_ops.so",
    srcs = glob(["ops/*.cc"]),
    deps = [
        "//tensorflow/contrib/persona/kernels:persona_kernels",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["persona_ops"],
)

#tf_gen_op_wrapper_py(
    #name = "persona_ops",
    #deps = [":persona_ops_op_lib"],
#)

tf_kernel_library(
    name = "persona_ops_kernels",
    deps = [
        "//tensorflow/contrib/persona/kernels:persona_kernels",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

py_library(
    name = "persona_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    data = [":python/ops/_persona_ops.so"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)


filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
