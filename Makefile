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

# Include user-defined build configuration file
include options.cfg

# Compilers
#
# Automatically detect if the intel compilers are installed and use
# them, otherwise default to the GNU compilers
ifeq ($(FORCE_GCC),ON)
	# Forcing the use of GCC
	# C Compiler
	CC  = mpicc
	# C++ Compiler
	CXX = mpicxx
	# Fortran Compiler
	FC  = mpif90
else
	ifeq (,$(shell which mpiicc))
		# C Compiler
		CC  = mpicc
		# C++ Compiler
		CXX = mpicxx
		# Fortran Compiler
		FC  = mpif90
	else
		# C Compiler
		CC  = mpiicc
		# C++ Compiler
		CXX = mpiicpc
		# Fortran Compiler
		FC  = mpiifort
	endif
endif


# Compiler flags
#
ifeq ($(CC),mpicc)
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
#CFLAGS   += -std=c99
# C++ standard
CXXFLAGS += -std=c++11
# Header includes
CXXFLAGS += -I${INC_PATH}


# Defines
#
DFLAGS = -DNPY_NO_DEPRECATED_API
ifeq ($(USE_GESVF),ON)
	DFLAGS += -DUSE_LAPACK_DGESVD
endif
ifeq ($(USE_MKL),ON)
	DFLAGS += -DUSE_MKL
endif
ifeq ($(USE_FFTW),ON)
	DFLAGS += -DUSE_FFTW3
endif


# One rule to compile them all, one rule to find them,
# One rule to bring them all and in the compiler link them.
all: deps python install
	@echo ""
	@echo "pyLOM deployed successfully"

# Selection of vector_matrix kernel libraries
#
ifeq ($(USE_MKL),ON)
vector_matrix: mkl

else
vector_matrix: lapack openblas

endif

# Selection of fft kernel libraries
#
ifeq ($(USE_FFTW),ON)
fft: fftw nfft

else
fft: kissfft

endif

# General rule to build dependencies
#
deps: vector_matrix fft requirements


# Python
#
python: setup.py
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PYTHON} $< build_ext --inplace
	@echo "Python programs deployed successfully"

requirements: requirements.txt
	@${PIP} install -r $<

install: requirements python
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PIP} install --no-deps --use-pep517 .

install_dev: requirements python
	@CC="${CC}" CFLAGS="${CFLAGS} ${DFLAGS}" CXX="${CXX}" CXXFLAGS="${CXXFLAGS} ${DFLAGS}" LDSHARED="${CC} -shared" ${PIP} install --no-deps --use-pep517 -e .

package-build:
	@CFLAGS="${CFLAGS}" CXXFLAGS="${CXXFLAGS}" LDSHARED="${CC} -shared" ${PYTHON} -m build


# External libraries
#
lapack: Deps/lapack
	@bash $</install_lapack.sh "${LAPACK_VERS}" "${PWD}/$<" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"
openblas: Deps/lapack
	@bash $</install_openblas.sh "${OPENBLAS_VERS}" "${PWD}/$<" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"
mkl: Deps/oneAPI
	@bash $</install_mkl.sh "${PLATFORM}" "${ONEAPI_VERS}" "${PWD}/$</mkl" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"
kissfft: Deps/kissfft
	@bash $</install_kissfft.sh "${PLATFORM}" "${KISSFFT_VERS}" "${OPENMP_PARALL}" "${PWD}/$<" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"
fftw: Deps/fftw
	@bash $</install_fftw.sh "${PLATFORM}" "${FFTW_VERS}" "${OPENMP_PARALL}" "${PWD}/$<" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"
nfft: Deps/nfft
	@bash $</install_nfft.sh "${PLATFORM}" "${NFFT_VERS}" "${OPENMP_PARALL}" "${PWD}/$<" "${CC}" "${CFLAGS}" "${FC}" "${FFLAGS}"


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
	-@cd pyLOM; rm -rf *.o $(wildcard **/*.o)
	-@cd pyLOM; rm -rf *.pyc $(wildcard **/*.pyc)
	-@cd pyLOM; rm -rf __pycache__ utils/__pycache__ vmmath/__pycache__ inp_out/__pycache__
	-@cd pyLOM; rm -rf POD/__pycache__ POD/*.c POD/*.cpp POD/*.html
	-@cd pyLOM; rm -rf DMD/__pycache__ DMD/*.c DMD/*.cpp DMD/*.html
	-@cd pyLOM; rm -rf SPOD/__pycache__ SPOD/*.c SPOD/*.cpp SPOD/*.html
	-@cd pyLOM; rm -rf vmmath/*.c vmmath/*.cpp vmmath/*.html
	-@cd pyLOM; rm -rf inp_out/*.c inp_out/*.cpp inp_out/*.html

cleanall: clean
	-@rm -rf build
	-@cd pyLOM; rm vmmath/*.so inp_out/*.so POD/*.so DMD/*.so SPOD/*.so

ifeq ($(USE_MKL),ON)
uninstall_vector_matrix: uninstall_mkl 

else
uninstall_vector_matrix: uninstall_lapack 

endif

ifeq ($(USE_FFTW),ON)
uninstall_fft: uninstall_fftw uninstall_nfft

else
uninstall_fft: uninstall_kissfft

endif

uninstall_deps: uninstall_vector_matrix uninstall_fft

uninstall: cleanall uninstall_deps uninstall_python
	@${PIP} uninstall pyLowOrder
	-@rm -rf pyLowOrder.egg-info

uninstall_python:
	@${PIP} uninstall pyLowOrder
	-@rm -rf pyLowOrder.egg-info

uninstall_lapack: Deps/lapack/lib
	-@rm -rf Deps/lapack/include
	-@rm -rf Deps/lapack/lib
	-@rm -rf Deps/lapack/share

uninstall_mkl: Deps/oneAPI/mkl
	-@Deps/oneAPI/l_BaseKit_p_${ONEAPI_VERS}.sh -a --silent --action remove --eula accept --components intel.oneapi.lin.mkl.devel --install-dir $(shell pwd)/Deps/oneAPI
	-@rm -rf Deps/oneAPI/l_BaseKit_p_${ONEAPI_VERS}.sh Deps/oneAPI/mkl

uninstall_kissfft: Deps/kissfft/lib
	-@rm -rf Deps/kissfft/include
	-@rm -rf Deps/kissfft/lib

uninstall_fftw: Deps/fftw/lib
	-@rm -rf Deps/fftw/include
	-@rm -rf Deps/fftw/lib

uninstall_nfft: Deps/nfft/lib
	-@rm -rf Deps/nfft/include
	-@rm -rf Deps/nfft/lib
	-@rm -rf Deps/nfft/share
