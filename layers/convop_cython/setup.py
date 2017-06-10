from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("convop", ["convop.pyx"])]

setup(
  name = 'convop',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],
  ext_modules = ext_modules
)
