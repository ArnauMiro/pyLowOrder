#!/bin/bash
#
# SCRIPT to deploy the FFTW libraries compiled
# as static libraries
#
# Arnau Miro, BSC (2021)

VERS=${1}
INSTALL_PREFIX=${2}
CCOMPILER=${3}
CFLAGS=${4}
FCOMPILER=${5}
FFLAGS=${6}

# Web address (source)
TAR=fftw-${VERS}.tar.gz
SRC=https://www.fftw.org/${TAR}

# Check if the FFTW libraries have been deployed
# if not, compile
if test -f "${INSTALL_PREFIX}/lib/fftw.a"; then
    echo "FFTW already deployed!"
    echo "Skipping build..."
else
    echo "FFTW not deployed!"
    echo "Starting build..."

    echo "Version ${VERS}"
    echo "C compiler '${CCOMPILER}' with flags '${CFLAGS}'"
    echo "Fortran compiler '${FCOMPILER}' with flags '${FFLAGS}'"
    echo "Install path ${INSTALL_PREFIX}"

	# Clone repository and checkout version tag
	wget -O ${TAR} ${SRC}
	tar xvzf ${TAR}
	#cd fftw-${VERS}

	# Configure CMAKE build
	mkdir -p build_deps
	cd build_deps
	cmake ../fftw-${VERS} \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
		-DCMAKE_INSTALL_LIBDIR=$INSTALL_PREFIX/lib \
		-DENABLE_THREADS=ON -DENABLE_OPENMP=ON -DENABLE_AVX2=ON -DBUILD_SHARED_LIBS=OFF \
		-DCMAKE_C_COMPILER="${CCOMPILER}" -DCMAKE_C_FLAGS="${CFLAGS}" \
		-DCMAKE_Fortran_COMPILER="${FCOMPILER}" -DCMAKE_Fortran_FLAGS="${FFLAGS}"

	# Build
	make -j $(getconf _NPROCESSORS_ONLN)
	make install

	# Cleanup
	cd ..
	rm -rf fftw-${VERS}.tar.gz fftw-${VERS} build_deps
fi