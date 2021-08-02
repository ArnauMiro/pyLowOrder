#!/usr/bin/env python
#
# pyLOM, setup.
#
# Setup and cythonize code.
#
# Last rev: 07/07/2021
from __future__ import print_function, division

import os, sys, numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

with open('README.md') as f:
	readme = f.read()


## Libraries and includes
include_dirs  = ['pyLOM/POD/src','pyLOM/utils/src',np.get_include()]
extra_objects = []
libraries     = ['m']


## Select which libraries to use depending on the compilation options
if os.environ['USE_MKL'] == 'ON':
	# Link with Intel MKL using the intel compilers
	# this is the most performing option available
	mklroot        = 'Deps/oneAPI/mkl'
	include_dirs  += [os.path.join(mklroot,'latest/include')]
	extra_objects += [os.path.join(mklroot,'libmkl_intel.a' if os.environ['CC'] == 'icc' else 'libmkl_gcc.a')]
else:
	# Link with OpenBLAS which has a decent performance but is not
	# as fast as Intel MKL
	include_dirs  += ['Deps/lapack/include/openblas']
	extra_objects += ['Deps/lapack/lib/libopenblas.a']
	libraries     += ['gfortran']
	# Classical LAPACK & BLAS library has a very bad performance
	# but is left here for nostalgia
	#include_dirs  += ['Deps/lapack/include/']
	#extra_objects += ['Deps/lapack/lib/liblapacke.a','Deps/lapack/lib/liblapack.a','Deps/lapack/lib/libcblas.a','Deps/lapack/lib/libblas.a']
	#libraries     += ['gfortran']
	# FFTW
	include_dirs  += ['Deps/fftw/include']
	extra_objects += ['Deps/fftw/lib/libfftw3.a','Deps/fftw/lib/libfftw3_omp.a']


## Modules
Module_POD = Extension('pyLOM.POD.wrapper',
						sources       = ['pyLOM/POD/wrapper.pyx',
										 'pyLOM/POD/src/pod.c',
										 'pyLOM/utils/src/matrix.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs,
						extra_objects = extra_objects,
						libraries     = libraries,
					   )


## Decide which modules to compile
modules_list = [Module_POD]


## Main setup
setup(
	name        = 'pyLOM',
	version     = '1.0.0',
	ext_modules = cythonize(modules_list,
		language_level = str(sys.version_info[0]), # This is to specify python 3 synthax
		annotate       = True                      # This is to generate a report on the conversion to C code
	),
    long_description = readme,
    url              = 'https://github.com/ArnauMiro/UPM_BSC_LowOrder',
    packages         = find_packages(exclude=('Deps','Examples','Docs')),
	install_requires = ['numpy','matplotlib','cython','h5py','mpi4py']
)