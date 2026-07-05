py_library(
    name = "autotune_cache",
    srcs = [
        "__init__.py",
        "cache.py",
        "decorator.py",
        "export.py",
    ],
    data = glob(["configs/**/*.json"]),
    deps = [
        "//rtp_llm:gpu_info",
    ],
    visibility = ["//visibility:public"],
)
