load("//tensorflow:tensorflow.bzl", "tf_copts")

genrule(
    name = "antlr",
    srcs = [],
    visibility = ["//visibility:public"],
    outs = ["FilteringLexer.cpp", "Filtering.tokens", "FilteringLexer.h", "FilteringLexer.tokens", "FilteringVisitor.cpp", "FilteringVisitor.h", "FilteringBaseListener.cpp", "FilteringBaseListener.h", "FilteringListener.cpp", "FilteringListener.h", "FilteringBaseVisitor.cpp", "FilteringBaseVisitor.h", "FilteringParser.cpp", "FilteringParser.h" ],
    cmd = "java -jar antlr-4.7-complete.jar -Dlanguage=Cpp -listener -visitor ../Filtering.g4",
)