#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from distutils.core import setup
from Cython.Build import cythonize
import numpy
import glob

#add method to find all pyx files in all folders
def get_cython_modules():
    return

setup(ext_modules=cythonize("./BlochSolver/CBlochSolver/CQuantumSolvers/c_rotations/rotation_handler.pyx", 
 language_level=3), include_dirs=[numpy.get_include()])
