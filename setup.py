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


## Select which libraries to use depending on the compilation options
if os.environ['USE_MKL'] == 'ON':
	# Link with Intel MKL using the intel compilers
	# this is the most performing option available
	mklroot             = 'Deps/oneAPI/mkl'
	lapack_include_dir  = os.path.join(mklroot,'latest/include')
	lapack_extra_obj    = [os.path.join(mklroot,'libmkl_intel.a' if os.environ['CC'] == 'icc' else 'libmkl_gcc.a')]
	lapack_libraries    = ['m']
else:
	# Link with OpenBLAS which has a decent performance but is not
	# as fast as Intel MKL
	lapack_include_dir  = 'Deps/lapack/include/openblas'
	lapack_extra_obj    = ['Deps/lapack/lib/libopenblas.a']
	# Classical LAPACK & BLAS library has a very bad performance
	# but is left here for nostalgia
	#lapack_include_dir  = 'Deps/lapack/include/'
	#lapack_extra_obj    = ['Deps/lapack/lib/liblapacke.a','Deps/lapack/lib/liblapack.a','Deps/lapack/lib/libcblas.a','Deps/lapack/lib/libblas.a']
	lapack_libraries    = ['m','gfortran']


## Modules
Module_POD = Extension('pyLOM.POD.wrapper',
						sources       = ['pyLOM/POD/wrapper.pyx',
										 'pyLOM/POD/src/pod.c',
										 'pyLOM/cutils/matrix.c',
									    ],
						language      = 'c',
						include_dirs  = ['pyLOM/POD/src','pyLOM/cutils',lapack_include_dir,np.get_include()],
						extra_objects = lapack_extra_obj,
						libraries     = lapack_libraries,
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
	install_requires = ['numpy','matplotlib','cython','mpi4py']
)