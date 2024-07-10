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
SRC="https://registrationcenter-download.intel.com/akdlm/IRC_NAS/9a98af19-1c68-46ce-9fdd-e249240c7c42/l_BaseKit_p_${VERS}.sh"

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
	if [ "$PLATFORM" = "MN5_GPP" ]; then # MareNostrum5 GPP
		# MKL path
		MKL_INSTALL_DIR="/gpfs/apps/MN5/GPP/ONEAPI/${VERS}/mkl/latest"
		MKL_LIBRARIES="${MKL_INSTALL_DIR}/lib/intel64/"
		# Create install directory and copy includes
		mkdir -p ${INSTALL_PREFIX}
		cp -r $MKL_INSTALL_DIR/include ${INSTALL_PREFIX}
	elif [ "$PLATFORM" = "MN4" ]; then # MareNostrum4
		# MKL path
		MKL_INSTALL_DIR="/apps/INTEL/oneapi/${VERS}/mkl/latest"
		MKL_LIBRARIES="${MKL_INSTALL_DIR}/lib/intel64/"
		# Create install directory and copy includes
		mkdir -p ${INSTALL_PREFIX}
		cp -r $MKL_INSTALL_DIR/include ${INSTALL_PREFIX}
	elif [ "$PLATFORM" = "MN4_MKL" ]; then # MareNostrum4 (pre OneAPI)
		# MKL path
		MKL_INSTALL_DIR="/apps/INTEL/${VERS}/mkl"
		MKL_LIBRARIES="${MKL_INSTALL_DIR}/lib/intel64/"
		# Create install directory and copy includes
		mkdir -p ${INSTALL_PREFIX}
		cp -r $MKL_INSTALL_DIR/include ${INSTALL_PREFIX}
	elif [ "$PLATFORM" = "MN3" ]; then # MareNostrum3
		# MKL path
		MKL_INSTALL_DIR="/apps/ONEAPI/${VERS}/mkl/latest"
		MKL_LIBRARIES="${MKL_INSTALL_DIR}/lib/intel64/"
		# Create install directory and copy includes
		mkdir -p ${INSTALL_PREFIX}
		cp -r $MKL_INSTALL_DIR/include ${INSTALL_PREFIX}
	elif [ "$PLATFORM" = "VEGA" ]; then # VEGA
		# MKL path
		MKL_INSTALL_DIR="/ceph/hpc/software/intel/oneapi/mkl/${VERS}/"
		MKL_LIBRARIES="${MKL_INSTALL_DIR}/lib/intel64/"
		# Create install directory and copy includes
		mkdir -p ${INSTALL_PREFIX}
		cp -r $MKL_INSTALL_DIR/include ${INSTALL_PREFIX}
	elif [ "$PLATFORM" = "FT3" ]; then # Finisterre 3
		# MKL path
		MKL_INSTALL_DIR="/opt/cesga/2020/software/MPI/intel/${VERS}/impi/${VERS}/imkl/${VERS}/mkl"
		MKL_LIBRARIES="${MKL_INSTALL_DIR}/lib/intel64/"
		# Create install directory and copy includes
		mkdir -p ${INSTALL_PREFIX}
		cp -r $MKL_INSTALL_DIR/include ${INSTALL_PREFIX}
	elif [ "$PLATFORM" = "flexo" ]; then # flexo
		# MKL path
		MKL_INSTALL_DIR="/opt/ohpc/pub/compiler/intel/oneapi/mkl/${VERS}/"
		MKL_LIBRARIES="${MKL_INSTALL_DIR}/lib/intel64/"
		# Create install directory and copy includes
		mkdir -p ${INSTALL_PREFIX}
		cp -r $MKL_INSTALL_DIR/include ${INSTALL_PREFIX}
	else
		cd Deps/
		# MKL path
		MKL_LIBRARIES="${PWD}/oneAPI/mkl/latest/lib/intel64"
		MKL_INCLUDES="${PWD}/oneAPI/mkl/latest/include"
		# Download MKL binary
		wget -O ${PWD}/oneAPI/l_BaseKit_p_${VERS}.sh ${SRC}
		chmod +x ${PWD}/oneAPI/l_BaseKit_p_${VERS}.sh
		# Trigger build
		${PWD}/oneAPI/l_BaseKit_p_${VERS}.sh \
			-a \
			--silent \
			--action install \
			--eula accept \
			--components intel.oneapi.lin.mkl.devel \
			--install-dir ${PWD}/oneAPI
		# Symlink include folder
		ln -s ${MKL_INCLUDES} ${INSTALL_PREFIX}/include
	fi
	# Build intel and gcc MKL static libraries
	ar -rcT ${INSTALL_PREFIX}/libmkl_intel_omp.a ${MKL_LIBRARIES}/libmkl_core.a ${MKL_LIBRARIES}/libmkl_intel_thread.a ${MKL_LIBRARIES}/libmkl_intel_lp64.a
	ar -rcT ${INSTALL_PREFIX}/libmkl_intel.a ${MKL_LIBRARIES}/libmkl_core.a ${MKL_LIBRARIES}/libmkl_sequential.a ${MKL_LIBRARIES}/libmkl_intel_lp64.a
	ar -rcT ${INSTALL_PREFIX}/libmkl_gcc_omp.a ${MKL_LIBRARIES}/libmkl_core.a ${MKL_LIBRARIES}/libmkl_gnu_thread.a ${MKL_LIBRARIES}/libmkl_intel_lp64.a
	ar -rcT ${INSTALL_PREFIX}/libmkl_gcc.a ${MKL_LIBRARIES}/libmkl_core.a ${MKL_LIBRARIES}/libmkl_sequential.a ${MKL_LIBRARIES}/libmkl_intel_lp64.a
fi
