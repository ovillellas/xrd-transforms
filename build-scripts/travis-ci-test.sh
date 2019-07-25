#!/bin/sh

set -x -e

set -
. activate "${CONDA_ENV}"

set -x

(cd ..; make test)

