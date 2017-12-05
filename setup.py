import numpy
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension

setup(name="divmachines",
      version="0.1",
      description="A set of Factorization methods in pytorch",
      author="Daniel Morales",
      author_email="dnlmrls9@gmail.com",
      license="MIT",
      packages=find_packages(),
      install_requires=['torch', 'numpy', 'scipy', 'scikit-learn', 'joblib'],
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