#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Load system modules
spectre_load_sys_modules() {
    # module load oneapi/mpi/2021.10.0
    # module load gcc/11.2.0
    # module load oneapi/tbb/2021.10.0
    # module load oneapi/compiler-rt/2023.2.1
    # module load oneapi/mkl/2023.2.0
    # module load gsl
    # module load hdf5_18/1.8.20
    # module load boost/1.74.0
    # module load cmake/3.27.4
    # module load anaconda3/2023.07-2
    # module load mpich/ge/gcc/64/3.4.2
    module load oneapi/mpi/latest
}

# Unload system modules
spectre_unload_sys_modules() {
    module unload boost/1.74.0
    # module unload hdf5_18/1.8.20
    # module unload gsl
    # module unload oneapi/mkl/2023.2.0
    module unload oneapi/tbb/2021.10.0
    module unload oneapi/compiler-rt/2023.2.1
    module unload gcc/10.2.0
    module unload oneapi/mpi/2021.10.0
    module unload cmake/3.27.4
    # module unload anaconda3/2023.07-2
    module unload mpich/ge/gcc/64/3.3.2-192-cm9.1
}


spectre_setup_modules() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    "${SPECTRE_HOME}/support/Environments/setup/symmetry_gcc.sh" "$@"
    local ret=$?
    if [ "${ret}" -ne 0 ] ; then
        echo >&2
        echo "Module setup failed!" >&2
    fi
    return "${ret}"
}

spectre_unload_modules() {
    module unload spectre_python
    module unload charm_mpi
    module unload yaml-cpp
    # module unload spectre_boost
    module unload libxsmm
    module unload libsharp
    module unload catch2
    module unload brigand
    module unload blaze
    module unload slurm/19.05.8
    module unload jemalloc

    spectre_unload_sys_modules
}

spectre_load_modules() {
    spectre_load_sys_modules

    module load blaze
    module load brigand
    module load catch2
    module load libsharp
    module load libxsmm
    # module load spectre_boost
    module load yaml-cpp
    module load charm_mpi
    module load spectre_python
    module load slurm/slurm/21.08.8
    module load jemalloc
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    # -D USE_LD=ld - ld.gold seems to hang linking the main executables
        #   -D CMAKE_C_COMPILER=/cm/local/apps/gcc/10.2.0/bin/gcc \
        #   -D CMAKE_CXX_COMPILER=/cm/local/apps/gcc/10.2.0/bin/g++ \
        #   -D USE_LD=ld \
        #   -D SPECTRE_TEST_RUNNER="$(pwd)/bin/charmrun" \
        #   -D Python_EXECUTABLE=`which python3` \
    cmake -D CHARM_ROOT=/home/sma2/deps/charm1/charm/mpi-linux-x86_64-smp/ \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D MEMORY_ALLOCATOR=SYSTEM \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D LIBXSMM_ROOT=/home/sma2/deps/libxsmm1/libxsmm/ \
          -D BLAZE_ROOT=/home/sma2/deps/blaze/ \
           -D BRIGAND_ROOT=/home/sma2/deps/brigand/ \
           -D LIBSHARP_ROOT=/home/sma2/deps/libsharp/ \
            -D CATCH_INCLUDE_DIR=/home/sma2/deps/Catch2/include/catch2/ \
          "$@" \
          $SPECTRE_HOME
}
