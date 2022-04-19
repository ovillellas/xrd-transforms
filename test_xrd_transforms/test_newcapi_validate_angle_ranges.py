# tests for capi implementation of xy_to_gvec.
#
# these tests are more about making sure the function generates the
# right exceptions when incorrect arguments are used

from __future__ import absolute_import

from collections import namedtuple
import pytest
import numpy as np
from xrd_transforms._new_transforms_capi import makeOscillRotMat as make_sample_rmat

    
def test_correct_make_sample_rmat():
    result = make_sample_rmat(0.3, np.ones((5,)))
    assert result.shape == (5, 3, 3)


def test_incorrect_chi():
    # incorrect: None not allowed
    with pytest.raises(TypeError):
        make_sample_rmat(None, np.ones((5,)))
        

def test_incorrect_omes():
    # In the newcapi version of make_sample_rmat only an array of omegas
    # is currently accepted. The python wrapper handles conversion of a
    # single scalar to a vector of 1 element prior to calling.

    # incorrect: None not allowed
    with pytest.raises(ValueError):
        make_sample_rmat(0.3, None)

    # incorrect: too many dimensions
    with pytest.raises(ValueError):
        make_sample_rmat(0.3, np.ones((5, 5)))


