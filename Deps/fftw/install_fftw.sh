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
TAR=fftw-${VERS}.tar.gz
SRC=https://www.fftw.org/${TAR}

# Check if the FFTW libraries have been deployed
# if not, compile
if test -f "${INSTALL_PREFIX}/lib/libfftw3.a"; then
	echo "FFTW already deployed!"
	echo "Skipping build..."
else
	echo "FFTW not deployed!"
	echo "Starting build..."

	echo "Platform ${PLATFORM}"
	echo "Version ${VERS}"
	echo "C compiler '${CCOMPILER}' with flags '${CFLAGS}'"
	echo "Fortran compiler '${FCOMPILER}' with flags '${FFLAGS}'"
	echo "Install path ${INSTALL_PREFIX}"

	if [[ "$PLATFORM" = "MN5_GPP" ]]; then # MareNostrum5 GPP
		# Symlink to MN4 installation directory
		if [ "$CCOMPILER" = "mpicc" ]; then
			# GCC compiler
			ln -s "/gpfs/apps/MN5/GPP/FFTW/${VERS}/GCC/OPENMPI/include" "${INSTALL_PREFIX}/"
			ln -s "/gpfs/apps/MN5/GPP/FFTW/${VERS}/GCC/OPENMPI/lib" "${INSTALL_PREFIX}/"
		else
			# Intel compiler
			ln -s "/gpfs/apps/MN5/GPP/FFTW/${VERS}/INTEL/IMPI/include" "${INSTALL_PREFIX}/"
			ln -s "/gpfs/apps/MN5/GPP/FFTW/${VERS}/INTEL/IMPI/lib" "${INSTALL_PREFIX}/"
		fi
	elif [[ "$PLATFORM" == "MN3" || "$PLATFORM" == "MN4" || "$PLATFORM" == "MN4_MKL" ]]; then # MareNostrum4
		# Symlink to MN4 installation directory
		if [ "$CCOMPILER" = "mpicc" ]; then
			# GCC compiler
			ln -s "/apps/FFTW/${VERS}/GCC/OPENMPI/include" "${INSTALL_PREFIX}/"
			ln -s "/apps/FFTW/${VERS}/GCC/OPENMPI/lib" "${INSTALL_PREFIX}/"
		else
			# Intel compiler
			ln -s "/apps/FFTW/${VERS}/INTEL/IMPI/include" "${INSTALL_PREFIX}/"
			ln -s "/apps/FFTW/${VERS}/INTEL/IMPI/lib" "${INSTALL_PREFIX}/"
		fi
	elif [[ "$PLATFORM" == "VEGA" ]]; then # VEGA
		# GCC compiler
		ln -s "/cvmfs/sling.si/modules/el7/software/FFTW/${VERS}-gompi-2020b/include" "${INSTALL_PREFIX}/"
		ln -s "/cvmfs/sling.si/modules/el7/software/FFTW/${VERS}-gompi-2020b/lib" "${INSTALL_PREFIX}/"
	elif [[ "$PLATFORM" == "FT3" ]]; then # Finisterre 3
		# Intel compiler
		ln -s "/opt/cesga/2020/software/MPI/intel/2021.3.0/impi/2021.3.0/fftw/${VERS}/include" "${INSTALL_PREFIX}/"
		ln -s "/opt/cesga/2020/software/MPI/intel/2021.3.0/impi/2021.3.0/fftw/${VERS}/lib" "${INSTALL_PREFIX}/"
	else
		cd Deps/
		# Clone repository and checkout version tag
		wget -O ${TAR} ${SRC}
		tar xvzf ${TAR}
		# Configure CMAKE build
		mkdir -p build_deps
		cd build_deps
		cmake ../fftw-${VERS} \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
			-DCMAKE_INSTALL_LIBDIR=$INSTALL_PREFIX/lib \
			-DENABLE_THREADS=${USE_OMP} -DENABLE_OPENMP=${USE_OMP} \
			-DENABLE_SSE=ON -DENABLE_SSE2=ON -DENABLE_AVX=ON -DENABLE_AVX2=ON \
			-DBUILD_SHARED_LIBS=OFF \
			-DCMAKE_C_COMPILER="${CCOMPILER}" -DCMAKE_C_FLAGS="${CFLAGS}" \
			-DCMAKE_REQUIRED_LIBRARIES="m"
		# Build
		make -j $(getconf _NPROCESSORS_ONLN)
		make install
		# Cleanup
		cd ..
		rm -rf fftw-${VERS}.tar.gz fftw-${VERS} build_deps
	fi
fi
