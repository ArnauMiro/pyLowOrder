#!/bin/bash
#
# SCRIPT to deploy the FFTW libraries compiled
# as static libraries
#
# Arnau Miro, BSC (2021)

PLATFORM=${1}
VERS=${2}
USE_OMP=${3}
INSTALL_PREFIX=${4}
CCOMPILER=${5}
CFLAGS=${6}
FCOMPILER=${7}
FFLAGS=${8}

# Web address (source)
DIR=kissfft-${VERS}
SRC=https://github.com/mborgerding/kissfft.git

# Check if the KISSFFT libraries have been deployed
# if not, compile
if test -f "${INSTALL_PREFIX}/lib/libkissfft-double.a"; then
	echo "KISSFFT already deployed!"
	echo "Skipping build..."
else
	echo "KISSFFT not deployed!"
	echo "Starting build..."

	echo "Platform ${PLATFORM}"
	echo "Version ${VERS}"
	echo "C compiler '${CCOMPILER}' with flags '${CFLAGS}'"
	echo "Fortran compiler '${FCOMPILER}' with flags '${FFLAGS}'"
	echo "Install path ${INSTALL_PREFIX}"

	cd Deps/kissfft
	# Clone repository and checkout version tag
	git clone ${SRC} ${DIR}
	cd ${DIR}
	git checkout ${VERS}
	cd ..
	# Configure CMAKE build
	mkdir -p build_deps
	cd build_deps
	cmake ../${DIR} \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
		-DCMAKE_INSTALL_LIBDIR=$INSTALL_PREFIX/lib \
		-DCMAKE_INSTALL_INCLUDEDIR=$INSTALL_PREFIX/include \
		-DCMAKE_CXX_COMPILER_WORKS=ON \
		-DKISSFFT_DATATYPE=double \
		-DBUILD_SHARED_LIBS=OFF -DKISSFFT_STATIC=ON \
		-DKISSFFT_OPENMP=${USE_OMP} \
		-DKISSFFT_TEST=OFF -DKISSFFT_TOOLS=OFF -DKISSFFT_USE_ALLOCA=OFF -DKISSFFT_PKGCONFIG=OFF \
		-DCMAKE_C_COMPILER="${CCOMPILER}" -DCMAKE_C_FLAGS="${CFLAGS}" 
	# Build
	make -j $(getconf _NPROCESSORS_ONLN)
	make install
	# Cleanup
	cd ..
	rm -rf ${DIR} build_deps
fi
