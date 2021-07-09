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


## Modules
Module_POD = Extension('pyLOM.POD.wrapper',
						sources       = ['pyLOM/POD/wrapper.pyx',
										 'pyLOM/POD/src/pod.c',
										 'pyLOM/mat_math/mat_math.c',
									    ],
						language      = 'c',
						include_dirs  = ['pyLOM/POD/src','pyLOM/mat_math','Deps/lapack/include/',np.get_include()],
						extra_objects = ['Deps/lapack/lib/liblapacke.a','Deps/lapack/lib/liblapack.a','Deps/lapack/lib/libcblas.a','Deps/lapack/lib/libblas.a'],
						libraries     = ['m','gfortran'],
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