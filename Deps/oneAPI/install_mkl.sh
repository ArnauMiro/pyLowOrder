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

    echo "Version ${VERS}"
    echo "C compiler '${CCOMPILER}' with flags '${CFLAGS}'"
    echo "Fortran compiler '${FCOMPILER}' with flags '${FFLAGS}'"
    echo "Install path ${INSTALL_PREFIX}"

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

    # Build intel and gcc MKL static libraries
    ar -rcT ${INSTALL_PREFIX}/libmkl_intel.a ${PWD}/Deps/oneAPI/mkl/latest/lib/intel64/libmkl_core.a ${PWD}/Deps/oneAPI/mkl/latest/lib/intel64/libmkl_intel_thread.a ${PWD}/Deps/oneAPI/mkl/latest/lib/intel64/libmkl_intel_lp64.a
    ar -rcT ${INSTALL_PREFIX}/libmkl_gcc.a ${PWD}/Deps/oneAPI/mkl/latest/lib/intel64/libmkl_core.a ${PWD}/Deps/oneAPI/mkl/latest/lib/intel64/libmkl_gnu_thread.a ${PWD}/Deps/oneAPI/mkl/latest/lib/intel64/libmkl_intel_lp64.a
fi