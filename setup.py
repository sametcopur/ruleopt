from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy
import platform

numpy_include_dir = numpy.get_include()

compile_args = []
if platform.system() == "Windows":
    compile_args = ["/O2", "/favor:ATOM", "/fp:fast"]
elif platform.system() == "Darwin":
    compile_args = ["-Ofast", "-funroll-loops"]
else:
    compile_args = ["-Ofast", "-march=native", "-funroll-loops"]

ext_modules = cythonize(
    [
        Extension(
            "ruleopt.aux_classes.aux_classes",
            ["ruleopt/aux_classes/aux_classes.pyx"],
            include_dirs=[numpy_include_dir],
            extra_compile_args=compile_args
            + ["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
            language="c",
        )
    ],
    language_level="3",
)

setup(
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)