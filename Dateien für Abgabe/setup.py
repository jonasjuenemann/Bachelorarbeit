from setuptools import setup
from Cython.Build import cythonize

"""Nötig für die Kompilierung der Cython Datei"""
setup(
    ext_modules=cythonize("GoLCython.pyx")
)

"""
Eine alternative Möglichkeit zum Kompilieren des Moduls wäre:

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("GameOfLifeCython", ["GoL.pyx"])
    ],
)
"""

"""
Befehl für die Kompilierung ist dann im Terminal: 

    python setup.py build_ext --inplace 
"""
