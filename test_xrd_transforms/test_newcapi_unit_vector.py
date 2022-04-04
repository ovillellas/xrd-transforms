# tests for capi implementation of unit_vector.
#
# this tests are more about making sure the function generates the
# right exceptions when incorrect arguments are used
#
# TODO: There are two different entries in the C-module for a single or
#       for a vector of vectors to be normalized row-wise.
#       In the current implementation the second is not needed as the first
#       one supports both use cases.


from __future__ import absolute_import

import pytest
import numpy as np
from xrd_transforms._new_transforms_capi import unitRowVector as unit_vector
from xrd_transforms._new_transforms_capi import unitRowVectors as unit_vectors

def test_single_dimension():
    arr = np.arange(0., 81.)
    result = unit_vector(arr)
    assert result.shape == (81,)

    with pytest.raises(ValueError):
        result = unit_vectors(arr) # this one expects 2 dimensions

    
def test_two_dimensions():
    arr = np.arange(0., 81.).reshape(9,9)
    result = unit_vector(arr)
    assert result.shape == (9, 9)

    result = unit_vectors(arr)
    assert result.shape == (9, 9)

    
def test_too_many_dimensions():
    arr = np.arange(0., 81.).reshape((3,3,9))

    with pytest.raises(ValueError):
        result = unit_vector(arr)

    with pytest.raises(ValueError):
        result = unit_vectors(arr)


def test_not_enough_dimensions():
    arr = np.arange(0., 81.)

    with pytest.raises(ValueError):
        result = unit_vector(arr[0]) # 0 dim array!

    with pytest.raises(ValueError):
        result = unit_vectors(arr[0]) # 0 dim array!


def test_none():
    with pytest.raises(ValueError):
        result = unit_vector(None)

    with pytest.raises(ValueError):
        result = unit_vectors(None)

        
def test_not_an_array():
    with pytest.raises(ValueError):
        result = unit_vector("foo bar baz")

    with pytest.raises(ValueError):
        result = unit_vectors("foo bar baz")
