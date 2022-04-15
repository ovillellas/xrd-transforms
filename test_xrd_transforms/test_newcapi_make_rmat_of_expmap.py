# tests for capi implementation of make_rmat_of_expmap.
#
# these tests are more about making sure the function generates the
# right exceptions when incorrect arguments are used

from __future__ import absolute_import

from collections import namedtuple
import pytest
import numpy as np
from xrd_transforms._new_transforms_capi import makeRotMatOfExpMap as make_rmat_of_expmap

    
def test_correct_make_rmat_of_expmap():
    result = make_rmat_of_expmap(np.zeros((3,)))
    assert result.shape == (3, 3)

def test_incorrect_expmap():
    # incorrect: None not allowed
    with pytest.raises(ValueError):
        make_rmat_of_expmap(None)

    # incorrect: too many dimensions
    with pytest.raises(ValueError):
        make_rmat_of_expmap(np.zeros((5, 3)))

    # incorrect: incorrect dimensions
    with pytest.raises(ValueError):
        make_rmat_of_expmap(np.zeros((2,)))
