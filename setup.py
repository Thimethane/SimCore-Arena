"""
Build the C++ proximity module.

Usage:
    python setup.py build_ext --inplace

The compiled .so is placed in the project root so both main.py and
benchmark.py can import it with a plain `import proximity_cpp`.
"""

from setuptools import setup, Extension
import pybind11

ext = Extension(
    name="proximity_cpp",
    sources=["cpp_module/proximity.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-O3", "-std=c++17", "-march=native", "-ffast-math"],
)

setup(
    name="proximity_cpp",
    version="1.0.0",
    ext_modules=[ext],
)
