from pathlib import Path

from setuptools import Extension, setup
import pybind11


root = Path(__file__).resolve().parents[2]

setup(
    name="daocp",
    version="0.1.0",
    ext_modules=[
        Extension(
            "daocp",
            [str(root / "interfaces/python/daocp.cpp")],
            include_dirs=[pybind11.get_include(), str(root), str(root/"blasfeo/include")],
            depends=[str(root / "src.c"), str(root / "lib/libblasfeo.a")],
            extra_objects=[str(root / "lib/libblasfeo.a")],
            language="c++",
            extra_compile_args=["-O3"],
        )
    ],
)
