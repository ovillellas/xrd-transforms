#!/bin/sh

set -x -e

set -
. activate "${CONDA_ENV}"

set -x
conda env list

make -C .. test

