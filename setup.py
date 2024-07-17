#!/usr/bin/env python
#
# pyLOM, setup.
#
# Setup and cythonize code.
#
# Last rev: 28/10/2021
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
		if options[linep[0].strip()] == 'ON':  options[linep[0].strip()] = True
		if options[linep[0].strip()] == 'OFF': options[linep[0].strip()] = False


## Set up compiler options and flags
CC  = 'mpicc'   if options['FORCE_GCC'] or not os.system('which icc > /dev/null') == 0 else 'mpiicc'
CXX = 'mpicxx'  if options['FORCE_GCC'] or not os.system('which icc > /dev/null') == 0 else 'mpiicpc'
FC  = 'mpifort' if options['FORCE_GCC'] or not os.system('which icc > /dev/null') == 0 else 'mpiifort'

CFLAGS   = ''
CXXFLAGS = ' -std=c++11'
FFLAGS   = ''
DFLAGS   = ' -DNPY_NO_DEPRECATED_API'
if options['USE_MKL']:   DFLAGS += ' -DUSE_MKL'
if options['USE_FFTW']:  DFLAGS += ' -DUSE_FFTW3'
if options['USE_GESVD']: DFLAGS += ' -DUSE_LAPACK_DGESVD'
if CC == 'mpicc':
	# Using GCC as a compiler
	CFLAGS   += ' -O0 -g -rdynamic -fPIC' if options['DEBUGGING'] else ' -O%s -ffast-math -fPIC' % options['OPTL']
	CXXFLAGS += ' -O0 -g -rdynamic -fPIC' if options['DEBUGGING'] else ' -O%s -ffast-math -fPIC' % options['OPTL']
	FFLAGS   += ' -O0 -g -rdynamic -fPIC' if options['DEBUGGING'] else ' -O%s -ffast-math -fPIC' % options['OPTL']
	# Vectorization flags
	if options['VECTORIZATION']:
		CFLAGS   += ' -march=native -ftree-vectorize'
		CXXFLAGS += ' -march=native -ftree-vectorize'
		FFLAGS   += ' -march=native -ftree-vectorize'
	# OpenMP flag
	if options['OPENMP_PARALL']:
		CFLAGS   += ' -fopenmp'
		CXXFLAGS += ' -fopenmp'
else:
	# Using GCC as a compiler
	CFLAGS   += ' -O0 -g -traceback -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
	CXXFLAGS += ' -O0 -g -traceback -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
	FFLAGS   += ' -O0 -g -traceback -fPIC' if options['DEBUGGING'] else ' -O%s -fPIC' % options['OPTL']
	# Vectorization flags
	if options['VECTORIZATION']:
		CFLAGS   += ' -x%s -mtune=%s' % (options['HOST'],options['TUNE'])
		CXXFLAGS += ' -x%s -mtune=%s' % (options['HOST'],options['TUNE'])
		FFLAGS   += ' -x%s -mtune=%s' % (options['HOST'],options['TUNE'])
	# OpenMP flag
	if options['OPENMP_PARALL']:
		CFLAGS   += ' -qopenmp'
		CXXFLAGS += ' -qopenmp'
		DFLAGS   += ' -DUSE_OMP'


## Set up environment variables
os.environ['CC']       = CC
os.environ['CXX']      = CXX
os.environ['CFLAGS']   = CFLAGS + DFLAGS
os.environ['CXXFLAGS'] = CXXFLAGS + DFLAGS
os.environ['LDSHARED'] = CC + ' -shared'


## Libraries and includes
include_dirs  = []
extra_objects = []
libraries     = ['m']

# OSX needs to also link with python3.8 for reasons...
if sys.platform == 'darwin': libraries += ['python3.8']

if options['USE_FFTW']:
	# NFFT
	include_dirs  += ['Deps/nfft/include']
	extra_objects += ['Deps/nfft/lib/libnfft3.a']
	# FFTW
	include_dirs  += ['Deps/fftw/include']
	extra_objects += ['Deps/fftw/lib/libfftw3.a']
	if options['OPENMP_PARALL']: extra_objects += ['Deps/fftw/lib/libfftw3_omp.a']
	DFLAGS        += ' -DUSE_FFTW3'
else:
	# KISSFFT
	include_dirs  += ['Deps/kissfft/include/kissfft']
	extra_objects += ['Deps/kissfft/lib/libkissfft-double.a']

# Select which libraries to use depending on the compilation options
if options['USE_MKL']:
	# Link with Intel MKL using the intel compilers
	# this is the most performing option available
	mklroot        = 'Deps/oneAPI/mkl'
	include_dirs  += [os.path.join(mklroot,'include')]
	if options['OPENMP_PARALL']:
		extra_objects += [os.path.join(mklroot,'libmkl_intel_thread.a' if CC == 'mpiicc' else 'libmkl_gcc_thread.a')]
	else:
		extra_objects += [os.path.join(mklroot,'libmkl_intel.a' if CC == 'mpiicc' else 'libmkl_gcc.a')]
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


## Modules
# vmmath module
Module_math = Extension('pyLOM.vmmath.wrapper',
						sources       = ['pyLOM/vmmath/wrapper.pyx',
										 'pyLOM/vmmath/src/vector_matrix.c',
										 'pyLOM/vmmath/src/averaging.c',
										 'pyLOM/vmmath/src/svd.c',
										 'pyLOM/vmmath/src/fft.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
# input output module
Module_IO_ensight  = Extension('pyLOM.inp_out.io_ensight',
						sources      = ['pyLOM/inp_out/io_ensight.pyx'],
						language     = 'c',
						include_dirs = [np.get_include()],
						libraries    = libraries,
					   )
# low-order modules
Module_POD = Extension('pyLOM.POD.wrapper',
						sources       = ['pyLOM/POD/wrapper.pyx',
										 'pyLOM/vmmath/src/vector_matrix.c',
										 'pyLOM/vmmath/src/averaging.c',
										 'pyLOM/vmmath/src/svd.c',
										 'pyLOM/vmmath/src/truncation.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
Module_DMD = Extension('pyLOM.DMD.wrapper',
						sources       = ['pyLOM/DMD/wrapper.pyx',
										 'pyLOM/vmmath/src/vector_matrix.c',
										 'pyLOM/vmmath/src/averaging.c',
										 'pyLOM/vmmath/src/svd.c',
										 'pyLOM/vmmath/src/truncation.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
Module_SPOD = Extension('pyLOM.SPOD.wrapper',
						sources       = ['pyLOM/SPOD/wrapper.pyx',
										 'pyLOM/vmmath/src/vector_matrix.c',
										 'pyLOM/vmmath/src/averaging.c',
										 'pyLOM/vmmath/src/svd.c',
										 'pyLOM/vmmath/src/fft.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )


## Decide which modules to compile
modules_list = [
	# Math module
	Module_math,
	# IO module
	Module_IO_ensight,
	# Low order algorithms
	Module_POD,Module_DMD,Module_SPOD
] if options['USE_COMPILED'] else []


## Main setup
setup(
	name             = 'pyLowOrder',
	version          = '1.3.6',
	author           = 'Benet Eiximeno, Beka Begiashvili, Arnau Miro, Eusebio Valero, Oriol Lehmkuhl',
	author_email     = 'benet.eiximeno@bsc.es, beka.begiashvili@alumnos.upm.es, arnau.mirojane@bsc.es, eusebio.valero@upm.es, oriol.lehmkuhl@bsc.es',
	maintainer       = 'Benet Eiximeno, Arnau Miro',
	maintainer_email = 'benet.eiximeno@bsc.es, arnau.mirojane@bsc.es',
	ext_modules      = cythonize(modules_list,
		language_level = str(sys.version_info[0]), # This is to specify python 3 synthax
		annotate       = True                      # This is to generate a report on the conversion to C code
	),
    long_description = readme,
    url              = 'https://github.com/ArnauMiro/pyLowOrder',
    packages         = find_packages(exclude=('Deps','Examples','Docs','Converters')),
	install_requires = ['numpy','matplotlib','cython>=3.0.0','h5py>=3.0.0','mpi4py','nfft']
)
