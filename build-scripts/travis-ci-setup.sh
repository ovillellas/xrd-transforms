#!/bin/sh

# install miniconda3... dependencies in tests are handled using conda.
set -x -e

echo "${TRAVIS_OS_NAME}"

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
    MINICONDA_PLATFORM_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
elif [ "${TRAVIS_OS_NAME}" = "osx" ]; then
    MINICONDA_PLATFORM_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
    echo "Unsupported platform \"$TRAVIS_OS_NAME\". Check matrix/scripts."
    exit 1
fi

wget $MINICONDA_PLATFORM_URL -O miniconda.sh

chmod u+x miniconda.sh
./miniconda.sh -b

# setup conda environment
PKG_SPEC="${PYTHON_SPEC} ${NUMPY_SPEC} ${NUMBA_SPEC} pytest"
echo $PKG_SPEC

conda remove --all -q -y -n "${CONDA_ENV}"
conda create -n "${CONDA_ENV}" -q -y ${PKG_SPEC}


. $HOME/miniconda3/bin/activate "${CONDA_ENV}"

# dump environment
echo "=========================  ENVIRONMENT  ========================="
conda env export
echo "================================================================="
