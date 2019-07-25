#!/bin/sh

set -x -e

. activate "${CONDA_ENV}"

pushd ..
make test
popd
