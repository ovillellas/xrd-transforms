# tests for capi implementation of make_beam_rmat.
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
from xrd_transforms._new_transforms_capi import makeEtaFrameRotMat as make_beam_rmat

def test_correct():
    result = make_beam_rmat(np.r_[1.0, 0.0, 0.0], np.r_[0.0, 1.0, 0.0])
    assert result.shape == (3, 3)

    
def test_incorrect_args():
    with pytest.raises(ValueError):
        result = make_beam_rmat(None, np.r_[0.0, 1.0, 0.0])

    with pytest.raises(ValueError):
        result = make_beam_rmat(np.r_[1.0, 0.0, 0.0], None)

    with pytest.raises(ValueError):
        result = make_beam_rmat(None, None)


def test_incorrect_dimensions():
    with pytest.raises(ValueError):
        result = make_beam_rmat(np.r_[1.0, 0.0, 0.0], np.r_[0.0, 1.0])

    with pytest.raises(ValueError):
        result = make_beam_rmat(np.r_[1.0, 0.0], np.r_[0.0, 1.0, 0.0])
        
    with pytest.raises(ValueError):
        result = make_beam_rmat(np.r_[1.0, 0.0], np.r_[0.0, 1.0])


def test_incorrect_dimensionality():
    with pytest.raises(ValueError):
        result = make_beam_rmat(np.r_[1.0, 0.0, 0.0], np.eye(3))

    with pytest.raises(ValueError):
        result = make_beam_rmat(np.eye(3), np.r_[0.0, 1.0, 0.0])
        
    with pytest.raises(ValueError):
        result = make_beam_rmat(np.eye(3), np.eye(3))

