#!/bin/bash

set -x -e

echo "${CONDA_ENV}"

echo $PATH
. activate "${CONDA_ENV}"

echo $PATH
conda env list

make test

