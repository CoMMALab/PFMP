# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os, subprocess, sys

PKG_NAME = "geofikinpy"

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

def find_eigen_include():
    # 1) Env override
    env = os.getenv("EIGEN3_INCLUDE_DIR")
    if env and os.path.isdir(env):
        return env
    # 2) Common locations
    candidates = [
        "/usr/include/eigen3",                # Debian/Ubuntu
        "/usr/local/include/eigen3",
        "/opt/homebrew/include/eigen3",       # macOS ARM (Homebrew)
        "/usr/local/opt/eigen/include/eigen3" # macOS Intel (Homebrew)
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # 3) Try brew (macOS)
    try:
        prefix = subprocess.check_output(["brew", "--prefix", "eigen"], text=True).strip()
        inc = os.path.join(prefix, "include", "eigen3")
        if os.path.isdir(inc):
            return inc
    except Exception:
        pass
    raise RuntimeError(
        "Eigen headers not found. Install libeigen3-dev (Linux) or `brew install eigen` (macOS), "
        "or set EIGEN3_INCLUDE_DIR=/path/to/eigen3"
    )

eigen_inc = find_eigen_include()

# Source paths (GeoFIK submodule lives inside the package folder)
core_src = os.path.join(PKG_NAME, "_core.cpp")
geofik_src = os.path.join(PKG_NAME, "geofik", "geofik.cpp")
geofik_inc = os.path.join(PKG_NAME, "geofik")

# Sanity checks (nice errors if submodule not initialized)
for path in [core_src, geofik_src]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing source file: {path} "
                                f"(did you run `git submodule update --init --recursive`?)")

ext_modules = [
    Extension(
        f"{PKG_NAME}._core",
        sources=[core_src, geofik_src],
        include_dirs=[
            get_pybind_include(),    # pybind11
            eigen_inc,               # Eigen
            geofik_inc,              # GeoFIK headers
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17", "-fvisibility=hidden"],
    )
]

class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        if sys.platform == "darwin":
            # libc++ for clang on macOS
            for ext in self.extensions:
                ext.extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]
        # Windows note: you’ll need Eigen path via EIGEN3_INCLUDE_DIR and MSVC will ignore -std flag.
        super().build_extensions()

setup(
    name=PKG_NAME,
    version="0.0.4",
    description="GeoFIK minimal binding (pybind11 + Eigen)",
    packages=[PKG_NAME],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    install_requires=["pybind11>=2.10", "numpy"],
    zip_safe=False,
)
