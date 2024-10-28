#!/bin/bash
#
# SCRIPT to deploy the NFFT libraries compiled
# as static libraries
#
# Arnau Miro, BSC (2021)

PLATFORM=${1}
VERS=${2}
USE_OMP=${3}
INSTALL_PREFIX=${4}
CCOMPILER=${5}
CFLAGS="${6}"
FCOMPILER=${7}
FFLAGS="${8}"

# Web address (source)
DIR=nfft-${VERS}
TAR=${DIR}.tar.gz
SRC=https://www-user.tu-chemnitz.de/~potts/nfft/download/${TAR}

# Check if the FFTW libraries have been deployed
# if not, compile
if test -f "${INSTALL_PREFIX}/lib/libnfft3.a"; then
	echo "NFFT already deployed!"
	echo "Skipping build..."
else
	echo "NFFT not deployed!"
	echo "Starting build..."

	echo "Version ${VERS}"
	echo "C compiler '${CCOMPILER}' with flags '${CFLAGS}'"
	echo "Fortran compiler '${FCOMPILER}' with flags '${FFLAGS}'"
	echo "Install path ${INSTALL_PREFIX}"

	# Clone repository and checkout version tag
	cd Deps/
	if [ ! -f "$${TAR}" ]; then
		wget --no-check-certificate -O ${TAR} ${SRC}
	fi
	tar xvzf ${TAR}
	# Configure build single
	cd ${DIR}
	if [ "$USE_OMP" = "ON" ]; then
		./configure --prefix=${INSTALL_PREFIX} \
					--enable-openmp \
					--enable-single \
					--with-fftw3=${INSTALL_PREFIX}/../fftw \
					CC="${CCOMPILER}" CFLAGS="${CFLAGS}" LDFLAGS="-lm"
	else
		./configure --prefix=${INSTALL_PREFIX} \
					--enable-single \
					--with-fftw3=${INSTALL_PREFIX}/../fftw \
					CC="${CCOMPILER}" CFLAGS="${CFLAGS}" LDFLAGS="-lm"
	fi
	# Build
	make -j $(getconf _NPROCESSORS_ONLN)
	make install
	# Configure build double
	if [ "$USE_OMP" = "ON" ]; then
		./configure --prefix=${INSTALL_PREFIX} \
					--enable-openmp \
					--with-fftw3=${INSTALL_PREFIX}/../fftw \
					CC="${CCOMPILER}" CFLAGS="${CFLAGS}" LDFLAGS="-lm"
	else
		./configure --prefix=${INSTALL_PREFIX} \
					--with-fftw3=${INSTALL_PREFIX}/../fftw \
					CC="${CCOMPILER}" CFLAGS="${CFLAGS}" LDFLAGS="-lm"
	fi
	# Build
	make -j $(getconf _NPROCESSORS_ONLN)
	make install
	# Cleanup
	cd ..
	rm -rf ${TAR} ${DIR}
fi
