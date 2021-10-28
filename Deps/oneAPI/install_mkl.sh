#!/bin/bash
#
# SCRIPT to deploy the LAPACK libraries compiled
# as static libraries and including LAPACKE
#
# Arnau Miro, BSC (2021)

PLATFORM=${1}
VERS=${2}
INSTALL_PREFIX=${3}
CCOMPILER=${4}
CFLAGS=${5}
FCOMPILER=${6}
FFLAGS=${7}

# Web address (source)
SRC=https://registrationcenter-download.intel.com/akdlm/irc_nas/17977/l_BaseKit_p_${VERS}.sh

# Check if the MKL libraries have been deployed
# if not, compile
if test -f "${INSTALL_PREFIX}/libmkl_gcc.a"; then
	echo "MKL already deployed!"
	echo "Skipping build..."
else
	echo "MKL not deployed!"
	echo "Starting build..."

	echo "Platform ${PLATFORM}"
	echo "Version ${VERS}"
	echo "C compiler '${CCOMPILER}' with flags '${CFLAGS}'"
	echo "Fortran compiler '${FCOMPILER}' with flags '${FFLAGS}'"
	echo "Install path ${INSTALL_PREFIX}"

	# Decide according to the accepted platforms which
	# MKL version is used
	if [ "$PLATFORM" = "MN4" ]; then # MareNostrum4
		# MKL path
		MKL_INSTALL_DIR="/apps/INTEL/oneapi/${VERS}/mkl/latest"
		MKL_LIBRARIES="${MKL_INSTALL_DIR}/lib/intel64/"
		# Create install directory and copy includes
		mkdir -p ${INSTALL_PREFIX}
		cp -r $MKL_INSTALL_DIR/include ${INSTALL_PREFIX}
	else
		# MKL path
		MKL_LIBRARIES="${PWD}/Deps/oneAPI/mkl/latest/lib/intel64"
		# Download MKL binary
		wget -O ${PWD}/Deps/oneAPI/l_BaseKit_p_${VERS}.sh ${SRC}
		chmod +x ${PWD}/Deps/oneAPI/l_BaseKit_p_${VERS}.sh

		# Trigger build
		${PWD}/Deps/oneAPI/l_BaseKit_p_${VERS}.sh \
			-a \
			--silent \
			--action install \
			--eula accept \
			--components intel.oneapi.lin.mkl.devel \
			--install-dir ${PWD}/Deps/oneAPI
	fi
	# Build intel and gcc MKL static libraries
	ar -rcT ${INSTALL_PREFIX}/libmkl_intel_omp.a ${MKL_LIBRARIES}/libmkl_core.a ${MKL_LIBRARIES}/libmkl_intel_thread.a ${MKL_LIBRARIES}/libmkl_intel_lp64.a
	ar -rcT ${INSTALL_PREFIX}/libmkl_intel.a ${MKL_LIBRARIES}/libmkl_core.a ${MKL_LIBRARIES}/libmkl_sequential.a ${MKL_LIBRARIES}/libmkl_intel_lp64.a
	ar -rcT ${INSTALL_PREFIX}/libmkl_gcc_omp.a ${MKL_LIBRARIES}/libmkl_core.a ${MKL_LIBRARIES}/libmkl_gnu_thread.a ${MKL_LIBRARIES}/libmkl_intel_lp64.a
	ar -rcT ${INSTALL_PREFIX}/libmkl_gcc.a ${MKL_LIBRARIES}/libmkl_core.a ${MKL_LIBRARIES}/libmkl_sequential.a ${MKL_LIBRARIES}/libmkl_intel_lp64.a
fi