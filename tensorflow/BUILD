# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "ACKNOWLEDGMENTS",
])

# open source marker; do not remove

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
)

package_group(
    name = "internal",
    packages = ["//tensorflow/..."],
)

sh_binary(
    name = "swig",
    srcs = ["tools/swig/swig.sh"],
    data = glob(["tools/swig/**"]),
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "tensorflow_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python"],
)

# -------------------------------------------
# New rules should be added above this target.
# -------------------------------------------
cc_binary(
    name = "libtensorflow.so",
    linkshared = 1,
    deps = [
        "//tensorflow/core:tensorflow",
    ],
)
