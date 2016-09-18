SOURCES = [
    "dgif_lib.c",
    "egif_lib.c",
    "gif_font.c",
    "gif_hash.c",
    "gifalloc.c",
    "openbsd-reallocarray.c",
    "gif_err.c",
    "quantize.c",
]

HEADERS = [
    "gif_hash.h",
    "gif_lib.h",
    "gif_lib_private.h",
]

prefix_dir = "giflib-5.1.4/lib"

cc_library(
    name = "gif",
    srcs = [prefix_dir + "/" + source for source in SOURCES],
    hdrs = [prefix_dir + "/" + hdrs for hdrs in HEADERS],
    includes = [prefix_dir],
    defines = [
        "HAVE_CONFIG_H",
    ],
    visibility = ["//visibility:public"],
)
