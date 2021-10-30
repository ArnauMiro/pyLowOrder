#!/bin/bash
#
# SCRIPT to deploy the LAPACK libraries compiled
# as static libraries and including LAPACKE
#
# Arnau Miro, BSC (2021)

VERS=${1}
INSTALL_PREFIX=${2}
CCOMPILER=${3}
CFLAGS=${4}
FCOMPILER=${5}
FFLAGS=${6}

# Github repository (source)
SRC=https://github.com/Reference-LAPACK/lapack.git

# Check if the Lapack libraries have been deployed
# if not, compile
if test -f "${INSTALL_PREFIX}/lib/liblapack.a"; then
    echo "LAPACK already deployed!"
    echo "Skipping build..."
else
    echo "LAPACK not deployed!"
    echo "Starting build..."

    echo "Version ${VERS}"
    echo "C compiler '${CCOMPILER}' with flags '${CFLAGS}'"
    echo "Fortran compiler '${FCOMPILER}' with flags '${FFLAGS}'"
    echo "Install path ${INSTALL_PREFIX}"

	# Clone repository and checkout version tag
	cd Deps/
	git clone ${SRC} lapack-src
	cd lapack-src
	git checkout tags/v${VERS}
	cd ..

	# Configure CMAKE build
	mkdir -p build_deps
	cd build_deps
	cmake ../lapack-src \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
		-DCMAKE_INSTALL_LIBDIR=$INSTALL_PREFIX/lib \
		-DLAPACKE=ON -DCBLAS=ON \
		-DCMAKE_C_COMPILER="${CCOMPILER}" -DCMAKE_C_FLAGS="${CFLAGS}" \
		-DCMAKE_Fortran_COMPILER="${FCOMPILER}" -DCMAKE_Fortran_FLAGS="${FFLAGS}"

	# Build
	make -j $(getconf _NPROCESSORS_ONLN)
	make install

	# Cleanup
	cd ..
	rm -rf lapack-src build_deps
fi