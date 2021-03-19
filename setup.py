#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("./BlochSolver/CBlochSolver/CPerturbations/filters.pyx",  language_level=3), include_dirs=[numpy.get_include()])
