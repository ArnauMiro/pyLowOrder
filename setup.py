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


## Read compilation options
options = {}
with open('options.cfg') as f:
	for line in f.readlines():
		if '#' in line or len(line) == 1: continue # Skip comment
		linep = line.split('=')
		options[linep[0].strip()] = linep[1].strip()


## Libraries and includes
include_dirs  = ['pyLOM/POD/src','pyLOM/DMD/src','pyLOM/utils/src',np.get_include()]
extra_objects = []
libraries     = ['m']
# OSX needs to also link with python3.8 for reasons...
if sys.platform == 'darwin': libraries += ['python3.8']


## Select which libraries to use depending on the compilation options
if options['USE_MKL'] == 'ON':
	# Link with Intel MKL using the intel compilers
	# this is the most performing option available
	mklroot        = 'Deps/oneAPI/mkl'
	include_dirs  += [os.path.join(mklroot,'latest/include')]
	if options['OPENMP_PARALL'] == 'ON':
		extra_objects += [os.path.join(mklroot,'libmkl_intel_thread.a' if os.environ['CC'] == 'mpiicc' else 'libmkl_gcc_thread.a')]
	else:
		extra_objects += [os.path.join(mklroot,'libmkl_intel.a' if os.environ['CC'] == 'mpiicc' else 'libmkl_gcc.a')]
else:
	# Link with OpenBLAS which has a decent performance but is not
	# as fast as Intel MKL
	include_dirs  += ['Deps/lapack/include/openblas']
	extra_objects += ['Deps/lapack/lib/libopenblas.a']
	libraries     += ['gfortran',]
	# Classical LAPACK & BLAS library has a very bad performance
	# but is left here for nostalgia
	#include_dirs  += ['Deps/lapack/include/']
	#extra_objects += ['Deps/lapack/lib/liblapacke.a','Deps/lapack/lib/liblapack.a','Deps/lapack/lib/libcblas.a','Deps/lapack/lib/libblas.a']
	#libraries     += ['gfortran']
	# FFTW
	include_dirs  += ['Deps/fftw/include']
	extra_objects += ['Deps/fftw/lib/libfftw3.a']
	if options['OPENMP_PARALL'] == 'ON': extra_objects += ['Deps/fftw/lib/libfftw3_omp.a']


## Modules
Module_matrix = Extension('pyLOM.utils.matrix',
						sources       = ['pyLOM/utils/matrix.pyx',
										 'pyLOM/utils/src/matrix.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs,
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
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
Module_DMD = Extension('pyLOM.DMD.wrapper',
						sources       = ['pyLOM/DMD/wrapper.pyx',
										 'pyLOM/DMD/src/dmd.c',
										 'pyLOM/POD/src/pod.c',
										 'pyLOM/utils/src/matrix.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs,
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
Module_IO_ensight  = Extension('pyLOM.inp_out.io_ensight',
						sources      = ['pyLOM/inp_out/io_ensight.pyx'],
						language     = 'c',
						include_dirs = [np.get_include()],
						libraries    = libraries,
					   )


## Decide which modules to compile
modules_list = [
	Module_matrix,Module_POD,Module_DMD,
	Module_IO_ensight,
]


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
    packages         = find_packages(exclude=('Deps','Examples','Docs','Converters')),
	install_requires = ['numpy','matplotlib','cython','h5py','mpi4py']
)
