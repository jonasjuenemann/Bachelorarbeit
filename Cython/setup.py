from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("GoL.pyx")
)

"""
run setup.py build_ext --inplace 

python setup.py build_ext --inplace 

import helloworld
"""