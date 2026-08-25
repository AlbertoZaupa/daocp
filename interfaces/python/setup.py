import os
import platform
from pathlib import Path
import sys

from setuptools import Extension, setup
import pybind11


if sys.platform == "darwin":
    os.environ.setdefault("ARCHFLAGS", f"-arch {platform.machine()}")
    if platform.machine() == "arm64":
        os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")


root = Path(__file__).resolve().parents[2]

setup(
    name="daocp",
    version="0.1.0",
    ext_modules=[
        Extension(
            "daocp",
            [str(root / "interfaces/python/daocp.cpp")],
            include_dirs=[
                pybind11.get_include(),
                str(root / "include"),
                str(root / "blasfeo/include"),
            ],
            depends=[
                str(root / "include/daocp.h"),
                str(root / "lib/libdaocp.a"),
                str(root / "lib/libblasfeo.a"),
            ],
            extra_objects=[
                str(root / "lib/libdaocp.a"),
                str(root / "lib/libblasfeo.a"),
            ],
            language="c++",
            extra_compile_args=["-O3"],
        )
    ],
)
