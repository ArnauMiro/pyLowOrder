# Compile PYLOM
#   Compile with g++ or Intel C++ Compiler
#   Compile with the most aggressive optimization setting (O3)
#   Use the most pedantic compiler settings: must compile with no warnings at all
#
# The user may override any desired internal variable by redefining it via command-line:
#   make CXX=g++ [...]
#   make OPTL=-O2 [...]
#   make FLAGS="-Wall -g" [...]
#
# Arnau Miro 2021

# Optimization, host and CPU type
#
OPTL = 3
HOST = Host
TUNE = skylake


# Options
#
VECTORIZATION   = ON
OPENMP_PARALL   = ON
USE_MKL         = ON
FORCE_GCC       = OFF
DEBUGGING       = OFF


# Versions of the libraries
#
LAPACK_VERS   = 3.9.0
FFTW_VERS     = 3.3.9
OPENBLAS_VERS = 0.3.17
ONEAPI_VERS   = 2021.3.0.3219


# Compilers
#
# Automatically detect if the intel compilers are installed and use
# them, otherwise default to the GNU compilers
PYTHON = python3
PIP    = pip3
ifeq ($(FORCE_GCC),ON) 
	# Forcing the use of GCC
	# C Compiler
	CC = gcc
	# C++ Compiler
	CXX = g++
	# Fortran Compiler
	FC = gfortran
else
	ifeq (,$(shell which icc))
		# C Compiler
		CC = gcc
		# C++ Compiler
		CXX = g++
		# Fortran Compiler
		FC = gfortran
	else
		# C Compiler
		CC = icc
		# C++ Compiler
		CXX = icpc
		# Fortran Compiler
		FC = ifort
	endif
endif


# Compiler flags
#
ifeq ($(CC),gcc)
	# Using GCC as a compiler
	ifeq ($(DEBUGGING),ON)
		# Debugging flags
		CFLAGS   += -O0 -g -rdynamic -fPIC
		CXXFLAGS += -O0 -g -rdynamic -fPIC
		FFLAGS   += -O0 -g -rdynamic -fPIC
	else
		CFLAGS   += -O$(OPTL) -ffast-math -fPIC
		CXXFLAGS += -O$(OPTL) -ffast-math -fPIC
		FFLAGS   += -O$(OPTL) -ffast-math -fPIC
	endif
	# Vectorization flags
	ifeq ($(VECTORIZATION),ON)
		CFLAGS   += -march=native -ftree-vectorize
		CXXFLAGS += -march=native -ftree-vectorize
		FFLAGS   += -march=native -ftree-vectorize
	endif
	# OpenMP flag
	ifeq ($(OPENMP_PARALL),ON)
		CFLAGS   += -fopenmp -DUSE_OMP
		CXXFLAGS += -fopenmp -DUSE_OMP
	endif
else
	# Using INTEL as a compiler
	ifeq ($(DEBUGGING),ON)
		# Debugging flags
		CFLAGS   += -O0 -g -traceback -fPIC
		CXXFLAGS += -O0 -g -traceback -fPIC
		FFLAGS   += -O0 -g -traceback -fPIC
	else
		CFLAGS   += -O$(OPTL) -fPIC
		CXXFLAGS += -O$(OPTL) -fPIC
		FFLAGS   += -O$(OPTL) -fPIC
	endif
	# Vectorization flags
	ifeq ($(VECTORIZATION),ON)
		CFLAGS   += -x$(HOST) -mtune=$(TUNE)
		CXXFLAGS += -x$(HOST) -mtune=$(TUNE)
		FFLAGS   += -x$(HOST) -mtune=$(TUNE)
	endif
	# OpenMP flag
	ifeq ($(OPENMP_PARALL),ON)
		CFLAGS   += -qopenmp -DUSE_OMP
		CXXFLAGS += -qopenmp -DUSE_OMP
	endif
endif
# C standard
CFLAGS   += -std=c99
# C++ standard
CXXFLAGS += -std=c++11
# Header includes
CXXFLAGS += -I${INC_PATH}


# Defines
#
DFLAGS = -DNPY_NO_DEPRECATED_API
ifeq ($(USE_MKL),ON) 
	DFLAGS += -DUSE_MKL
endif


# One rule to compile them all, one rule to find them,
# One rule to bring them all and in the compiler link them.
all: deps python install
	@echo ""
	@echo "pyLOM deployed successfully"

ifeq ($(USE_MKL),ON) 
deps: mkl requirements

else
deps: lapack openblas fftw requirements

endif


# Python
#
python: setup.py
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" USE_MKL="${USE_MKL}" ${PYTHON} $< build_ext --inplace
	@echo "Python programs deployed successfully"

requirements: requirements.txt
	@${PIP} install -r $<

install: requirements python
	@CC="${CC}" USE_MKL="${USE_MKL}" ${PIP} install .

install_dev: requirements python
	@CC="${CC}" USE_MKL="${USE_MKL}" ${PIP} install -e .


# External libraries
#
lapack: Deps/lapack
	@bash $</install_lapack.sh "${LAPACK_VERS}" "${PWD}/$<" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"
openblas: Deps/lapack
	@bash $</install_openblas.sh "${OPENBLAS_VERS}" "${PWD}/$<" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"
mkl: Deps/oneAPI
	@bash $</install_mkl.sh "${ONEAPI_VERS}" "${PWD}/$</mkl" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"
fftw: Deps/fftw
	@bash $</install_fftw.sh "${FFTW_VERS}" "${PWD}/$<" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"


# Generic object makers
#
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $< $(DFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(DFLAGS)

%.o: %.f
	$(FC) $(FFLAGS) -c -o $@ $< $(DFLAGS)

%.o: %.f90
	$(FC) $(FFLAGS) -c -o $@ $< $(DFLAGS)


# Clean
#
clean:
	-@cd pyLOM; rm -f *.o $(wildcard **/*.o)
	-@cd pyLOM; rm -f *.pyc $(wildcard **/*.pyc)
	-@cd pyLOM; rm -rf __pycache__ POD/__pycache__ utils/__pycache__
	-@cd pyLOM; rm -f POD/*.c POD/*.cpp POD/*.html 
	-@cd pyLOM; rm -f utils/*.c utils/*.cpp utils/*.html 

cleanall: clean
	-@rm -rf build
	-@cd pyLOM; rm POD/*.so utils/*.so

ifeq ($(USE_MKL),ON) 
uninstall_deps: uninstall_mkl

else
uninstall_deps: uninstall_lapack uninstall_fftw

endif

uninstall: cleanall uninstall_deps uninstall_python
	@${PIP} uninstall pyLOM
	-@rm -rf pyLOM.egg-info

uninstall_python: 
	@${PIP} uninstall pyLOM
	-@rm -rf pyLOM.egg-info

uninstall_lapack: Deps/lapack/lib
	-@rm -rf Deps/lapack/include
	-@rm -rf Deps/lapack/lib
	-@rm -rf Deps/lapack/share

uninstall_mkl: Deps/oneAPI/l_BaseKit_p_${ONEAPI_VERS}.sh
	-@$< -a --silent --action remove --eula accept --components intel.oneapi.lin.mkl.devel --install-dir $(shell pwd)/Deps/oneAPI
	-@rm -rf $< Deps/oneAPI/mkl

uninstall_fftw: Deps/fftw/lib
	-@rm -rf Deps/fftw/include
	-@rm -rf Deps/fftw/lib