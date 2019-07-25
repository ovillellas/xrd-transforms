#!/bin/sh

. activate $CONDA_ENV

pushd ..
make test
popd
