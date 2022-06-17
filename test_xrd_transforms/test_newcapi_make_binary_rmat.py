# tests for capi implementation of make_binary_rmat
#
# these tests are more about making sure the function generates the
# right exceptions when incorrect arguments are used

from __future__ import absolute_import

from collections import namedtuple
import pytest
import numpy as np
from xrd_transforms._new_transforms_capi import makeBinaryRotMat as make_binary_rmat

    
def test_correct_make_binary_rmap():
    result = make_binary_rmat(np.ones((3,)))
    assert result.shape == (3, 3)

def test_incorrect_axis():
    # incorrect: None not allowed
    with pytest.raises(ValueError):
        make_binary_rmat(None)

    # incorrect: too many dimensions
    with pytest.raises(ValueError):
        make_binary_rmat(np.zeros((5, 3)))

    # incorrect: incorrect dimensions
    with pytest.raises(ValueError):
        make_binary_rmat(np.zeros((2,)))
