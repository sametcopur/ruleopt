# setup.py
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np
import platform

# --- scrub CPython's stray -LModules/_hacl on macOS/Python 3.13+ ---
try:
    from distutils import sysconfig as dist_sysconfig  # still bundled with setuptools
    cfg = dist_sysconfig.get_config_vars()
    for key in ("LDFLAGS", "LDSHARED"):
        val = cfg.get(key)
        if val and "Modules/_hacl" in val:
            cfg[key] = val.replace("-LModules/_hacl", "")
except Exception:
    # Non-fatal: fall back silently if anything changes upstream
    pass

numpy_include_dir = np.get_include()
sysname = platform.system()

if sysname == "Windows":
    extra_compile_args = ["/O2", "/GL", "/Gw", "/Gy", "/fp:fast", "/DNDEBUG"]
    extra_link_args    = ["/LTCG"]
elif sysname == "Darwin":
    extra_compile_args = ["-O3", "-flto", "-fvisibility=hidden", "-DNDEBUG"]
    extra_link_args    = ["-flto"]
else:
    extra_compile_args = ["-O3", "-flto", "-fvisibility=hidden", "-fPIC", "-DNDEBUG"]
    extra_link_args    = ["-flto"]

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

extensions = [
    Extension(
        "ruleopt.aux_classes.aux_classes",
        ["ruleopt/aux_classes/aux_classes.pyx"],
        include_dirs=[numpy_include_dir],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
    Extension(
        "ruleopt.solver.solver_utils",
        ["ruleopt/solver/solver_utils.pyx"],
        include_dirs=[numpy_include_dir],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
]

ext_modules = cythonize(
    extensions,
    language_level=3,
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "nonecheck": False,
        "cdivision": True,
        "infer_types": True,
        "embedsignature": False,
    },
)

setup(
    name="ruleopt",
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    zip_safe=False,
)
