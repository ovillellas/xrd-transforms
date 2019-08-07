#!/bin/bash

# install miniconda3... dependencies in tests are handled using conda.
set -x -e

echo "${TRAVIS_OS_NAME}"

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
    MINICONDA_PLATFORM_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
elif [ "${TRAVIS_OS_NAME}" = "osx" ]; then
    MINICONDA_PLATFORM_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
elif [ "${TRAVIS_OS_NAME}" = "windows" ]; then
    MINICONDA_PLATFORM_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
else
    echo "Unsupported platform \"$TRAVIS_OS_NAME\". Check matrix/scripts."
    exit 1
fi

if [ "${TRAVIS_OS_NAME}" != "windows" ]; then
    # unix-like
    wget $MINICONDA_PLATFORM_URL -O miniconda.sh

    chmod u+x miniconda.sh
    ./miniconda.sh -b
else
    # windows
    wget $MINICONDA_PLATFORM_URL -O miniconda.exe

    cmd /c miniconda.exe /S /D=%UserProfile%/Miniconda3
fi

# setup conda environment
PKG_SPEC="${PYTHON_SPEC} ${NUMPY_SPEC} ${NUMBA_SPEC} pytest"
echo $PKG_SPEC

conda remove --all -q -y -n "${CONDA_ENV}"
conda create -n "${CONDA_ENV}" -q -y ${PKG_SPEC}

set -

echo $PATH
. activate $CONDA_ENV
echo $PATH

# dump environment
echo "=========================  ENVIRONMENT  ========================="
conda env export
echo "================================================================="
