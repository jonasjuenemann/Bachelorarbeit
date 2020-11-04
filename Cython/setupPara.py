from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("GoLPara",
              ["GoLPara.pyx"],
              libraries=["m"],
              extra_compile_args=["-openmp"],
              extra_link_args=['-openmp']
              )
]

setup(
    name="GoLPara",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)

"""
run setupPara.py build_ext --inplace 
"""