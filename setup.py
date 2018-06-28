from setuptools import setup
from numpy.distutils.core import setup, Extension
import os

ext = Extension(name = 'flib',
                sources = [os.path.join(os.getcwd(),'climate','pySCM','flib.F90')])                


setup(name='climate',

      version='1.0',

      description='A package for Physics 201 term project on climate',

      url='http://github.com/mrowan137/climate',

      author='mrowan137',

      author_email='mrowan137@gmail.com',

      license='GPLv3',

      packages=['climate'],

      install_requires=['numpy'],
      
      test_suite='nose.collector',
      
      tests_require=['nose'],

      ext_modules = [ext]
)
