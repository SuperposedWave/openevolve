from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "_linear_code_native",
            ["examples/linear_code_binary_search/_linear_code_native.c"],
            extra_compile_args=["-O3"],
        )
    ]
)  # Most config in pyproject.toml
