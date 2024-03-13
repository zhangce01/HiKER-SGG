from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(name="bbox", ext_modules=cythonize('bbox.pyx'), include_dirs=[numpy.get_include()])