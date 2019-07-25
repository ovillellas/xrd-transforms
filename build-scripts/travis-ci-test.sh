#!/bin/sh

set -x -e

echo "${CONDA_ENV}"


. $HOME/miniconda3/bin/activate "${CONDA_ENV}"

conda env list

set -x

pwd

ls

make test

