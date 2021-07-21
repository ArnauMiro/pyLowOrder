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
FORCE_GCC       = OFF
DEBUGGING       = OFF

# Versions of the libraries
#
LAPACK_VERS   = 3.9.0
OPENBLAS_VERS = 0.3.17

# Compilers
#
# Automatically detect if the intel compilers are installed and use
# them, otherwise default to the GNU compilers
PYTHON = python
PIP    = pip
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


# One rule to compile them all, one rule to find them,
# One rule to bring them all and in the compiler link them.
all: deps python install
	@echo ""
	@echo "pyLOM deployed successfully"

deps: lapack openblas requirements

# Python
#
python: setup.py
	@CFLAGS="${CFLAGS} ${DFLAGS}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" ${PYTHON} $< build_ext --inplace
	@echo "Python programs deployed successfully"

requirements: requirements.txt
	@${PIP} install -r $<

install: requirements python
	@${PIP} install .

install_dev: requirements python
	@${PIP} install -e .


# External libraries
#
lapack: Deps/lapack
	@bash $</install_lapack.sh "${LAPACK_VERS}" "${PWD}/$<" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"
openblas: Deps/lapack
	@bash $</install_openblas.sh "${OPENBLAS_VERS}" "${PWD}/$<" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"


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
	-@cd pyLOM; rm -rf __pycache__ 
	-@cd pyLOM; rm -f wrapper.c wrapper.cpp wrapper.html 

cleanall: clean
	-@rm -rf build
	-@cd pyLOM; rm POD/*.so

uninstall: cleanall
	@${PIP} uninstall pyLOM
	-@rm -rf pyLOM.egg-info

uninstall_lapack:
	-@rm -rf Deps/lapack/include
	-@rm -rf Deps/lapack/lib