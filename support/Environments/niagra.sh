#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Load system modules
spectre_load_sys_modules() {
    module load NiaEnv/2019b
    # module load intel/2019u4
    # module load gcc/13.2.0
    # module load intel/2020u2
    module load cmake
    #module load impi/19.0.9
    module load gcc/12.2.0
    # module load mkl
    # module load hdf5/1.8.21
    # module load intelmpi/2020u2
    module load boost/1.78.0
    # module load gsl/2.7
    module load python/3.9.8
}

# Unload system modules
spectre_unload_sys_modules() {
    module unload boost/1.78.0
    module unload hdf5
    module unload gsl/2.7
    module unload mkl
    module unload cmake
    module unload gcc/13.2.0
    module unload NiaEnv/2019b
}


spectre_setup_modules() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    "${SPECTRE_HOME}/support/Environments/setup/niagra.sh" "$@"
    local ret=$?
    if [ "${ret}" -ne 0 ] ; then
        echo >&2
        echo "Module setup failed!" >&2
    fi
    return "${ret}"
}

spectre_unload_modules() {
    #module unload spectre_python
    #module unload charm_mpi
    #module unload yaml-cpp
    #module unload spectre_boost
    #module unload libxsmm
    #module unload libsharp
    #module unload catch
    #module unload brigand
    #module unload blaze

    spectre_unload_sys_modules
}

spectre_load_modules() {
    spectre_load_sys_modules

    module load blaze
    module load brigand
    module load catch
    module load libsharp
    module load libxsmm
    module load spectre_boost
    module load yaml-cpp
    module load charm_mpi
    module load spectre_python
    module load gsl
    module load hdf5
    module load catch2
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    # -D USE_LD=ld - ld.gold seems to hang linking the main executables
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D MEMORY_ALLOCATOR=SYSTEM \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D Python_EXECUTABLE=`which python3` \
          -D USE_LD=ld \
          -D SPECTRE_TEST_RUNNER="$(pwd)/bin/charmrun" \
          "$@" \
          $SPECTRE_HOME
}
