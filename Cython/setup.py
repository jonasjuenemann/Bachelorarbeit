from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("GameOfLifeCython", ["GoL.pyx"],
                             include_dirs=[np.get_include()]),
    ],
)


"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("GoL.pyx")
)


run setup.py build_ext --inplace 

python setup.py build_ext --inplace 

import helloworld
"""