import numpy
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension

USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.c'


extensions = [Extension("divmachines.mf.fast.fast_inner",
                            ["divmachines/mf/fast/fast_inner"+ext],
                                include_dirs=[numpy.get_include()]),
              Extension('divmachines.fm.second_order.fast.second_order_fast_inner',
                            ["divmachines/fm/second_order/fast/second_order_fast_inner"+ext],
                                include_dirs=[numpy.get_include()])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(name="FactorizationPyTorch",
      version="0.1",
      description="A set of Factorization methods in pytorch",
      author="Daniel Morales",
      author_email="dnlmrls9@gmail.com",
      license="MIT",
      packages=find_packages(),
      ext_modules=extensions,
      install_requires=['torch', 'numpy'],
      zip_safe=False,
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Programming Language :: C',
          'Programming Language :: Python',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS']
)
