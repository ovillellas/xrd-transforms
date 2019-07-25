#!/bin/sh


# install miniconda3... dependencies in tests are handled using conda.
set -v -e

echo "${TRAVIS_OS_NAME}"

if [ "${TRAVIS_OS_NAME}" == "linux" ]; then
    MINICONDA_PLATFORM_URL = https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
elif [ "${TRAVIS_OS_NAME}" == "osx" ]; then
    MINICONDA_PLATFORM_URL = https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
    echo "Unsupported platform \"$TRAVIS_OS_NAME\". Check matrix/scripts."
    exit 1
fi

wget $MINICONDA_PLATFORM_URL -O miniconda.sh

chmod u+x miniconda.sh
./miniconda.sh -b

export PATH=$HOME/miniconda3/bin:$PATH

# setup conda environment
CONDA_ENV=travisci_test
CONDA_INSTALL="conda install -q -y"

conda remove --all -q -y -n $CONDA_ENV
conda create -n $CONDA_ENV -q -y $PYTHON_SPEC $NUMPY_SPEC $NUMBA_SPEC
conda activate $CONDA_ENV

# dump environment
echo "=========================  ENVIRONMENT  ========================="
conda env export
echo "================================================================="
