#!/usr/bin/env python
#
# pyLOM, setup.
#
# Setup and cythonize code.
#
# Last rev: 28/10/2021
from __future__ import print_function, division

import os, sys, numpy as np, mpi4py
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


## Read INIT file
with open('pyLOM/__init__.py') as f:
	for l in f.readlines():
		if '__version__' in l:
			__version__ = eval(l.split('=')[1].strip())


## Read README file
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
options['MODULES_COMPILED'] = options['MODULES_COMPILED'].lower().split(',')


## Set up compiler options and flags
ICC = 'icx' if 'ACC' in options['PLATFORM'] else 'icc'
CC  = 'mpicc'   if options['FORCE_GCC'] or not os.system('which %s > /dev/null'%ICC) == 0 else 'mpiicc'
CXX = 'mpicxx'  if options['FORCE_GCC'] or not os.system('which %s > /dev/null'%ICC) == 0 else 'mpiicpc'
FC  = 'mpifort' if options['FORCE_GCC'] or not os.system('which %s > /dev/null'%ICC) == 0 else 'mpiifort'

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

# OSX needs to also link with python for reasons...
if sys.platform == 'darwin': libraries += [f'python{sys.version_info[0]}.{sys.version_info[1]}']

if options['USE_FFTW']:
	# NFFT
	include_dirs  += ['Deps/nfft/include']
	extra_objects += ['Deps/nfft/lib/libnfft3.a','Deps/nfft/lib/libnfft3f.a']
	# FFTW
	include_dirs  += ['Deps/fftw/include']
	extra_objects += ['Deps/fftw/lib/libfftw3.a','Deps/fftw/lib/libfftw3f.a']
	if options['OPENMP_PARALL']: extra_objects += ['Deps/fftw/lib/libfftw3_omp.a','Deps/fftw/lib/libfftw3f_omp.a']
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
#	include_dirs  += ['Deps/lapack/include/']
#	extra_objects += ['Deps/lapack/lib/liblapacke.a','Deps/lapack/lib/liblapack.a','Deps/lapack/lib/libcblas.a','Deps/lapack/lib/libblas.a']
#	libraries     += ['gfortran']


## Modules
# vmmath module
Module_cfuncs     = Extension('pyLOM.vmmath.cfuncs',
						sources       = ['pyLOM/vmmath/cfuncs.pyx',
										 'pyLOM/vmmath/src/vector_matrix.c',
										 'pyLOM/vmmath/src/averaging.c',
										 'pyLOM/vmmath/src/svd.c',
										 'pyLOM/vmmath/src/fft.c',
										 'pyLOM/vmmath/src/geometric.c',
										 'pyLOM/vmmath/src/truncation.c',
										 'pyLOM/vmmath/src/stats.c',
										 'pyLOM/vmmath/src/regression.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
Module_maths     = Extension('pyLOM.vmmath.maths',
						sources       = ['pyLOM/vmmath/maths.pyx',
										 'pyLOM/vmmath/src/vector_matrix.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
Module_averaging = Extension('pyLOM.vmmath.averaging',
						sources       = ['pyLOM/vmmath/averaging.pyx',
										 'pyLOM/vmmath/src/averaging.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
Module_svd       = Extension('pyLOM.vmmath.svd',
						sources       = ['pyLOM/vmmath/svd.pyx',
										 'pyLOM/vmmath/src/vector_matrix.c',
										 'pyLOM/vmmath/src/svd.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
Module_fft       = Extension('pyLOM.vmmath.fft',
						sources       = ['pyLOM/vmmath/fft.pyx',
										 'pyLOM/vmmath/src/fft.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
Module_geometric = Extension('pyLOM.vmmath.geometric',
						sources       = ['pyLOM/vmmath/geometric.pyx',
										 'pyLOM/vmmath/src/geometric.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
Module_truncation = Extension('pyLOM.vmmath.truncation',
						sources       = ['pyLOM/vmmath/truncation.pyx',
										 'pyLOM/vmmath/src/vector_matrix.c',
										 'pyLOM/vmmath/src/truncation.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
Module_stats     = Extension('pyLOM.vmmath.stats',
						sources       = ['pyLOM/vmmath/stats.pyx',
										 'pyLOM/vmmath/src/stats.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )
Module_regression = Extension('pyLOM.vmmath.regression',
						sources       = ['pyLOM/vmmath/regression.pyx',
										 'pyLOM/vmmath/src/vector_matrix.c',
										 'pyLOM/vmmath/src/regression.c',
									    ],
						language      = 'c',
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
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
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
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
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
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
						include_dirs  = include_dirs + ['pyLOM/vmmath/src',np.get_include(),mpi4py.get_include()],
						extra_objects = extra_objects,
						libraries     = libraries,
					   )


## Build modules
# Math module
Module_Math  = [Module_cfuncs]
Module_Math += [Module_maths]      if 'math.maths'      in options['MODULES_COMPILED'] else []
Module_Math += [Module_averaging]  if 'math.averaging'  in options['MODULES_COMPILED'] else []
Module_Math += [Module_svd]        if 'math.svd'        in options['MODULES_COMPILED'] else []
Module_Math += [Module_fft]        if 'math.fft'        in options['MODULES_COMPILED'] else []
Module_Math += [Module_geometric]  if 'math.geometric'  in options['MODULES_COMPILED'] else []
Module_Math += [Module_truncation] if 'math.truncation' in options['MODULES_COMPILED'] else []
Module_Math += [Module_stats]      if 'math.stats'      in options['MODULES_COMPILED'] else []
Module_Math += [Module_regression] if 'math.regression' in options['MODULES_COMPILED'] else []
# ROM module
Module_ROM   = [Module_POD]  if 'rom.pod'  in options['MODULES_COMPILED'] else []
Module_ROM  += [Module_DMD]  if 'rom.dmd'  in options['MODULES_COMPILED'] else []
Module_ROM  += [Module_SPOD] if 'rom.spod' in options['MODULES_COMPILED'] else []


## Decide which modules to compile
modules_list = Module_Math + Module_ROM if options['USE_COMPILED'] else []


## Main setup
setup(
	name             = 'pyLowOrder',
	version          = __version__,
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
    packages         = find_packages(exclude=('Converters','Examples','Deps','Testsuite','Tools')),
	install_requires = ['numpy','matplotlib','cython>=3.0.0','h5py>=3.0.0','mpi4py>=4.0.0','nfft']
)
