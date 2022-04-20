# tests for newcapi implementation of rotate_vecs_about_axis.
#
# these tests are more about making sure the function generates the
# right exceptions when incorrect arguments are used

from __future__ import absolute_import

from collections import namedtuple
import pytest
import numpy as np
from xrd_transforms._new_transforms_capi import rotate_vecs_about_axis

    
def test_correct_single_axis_angle():
    # note, angle required as a 0 dim array.
    vecs = np.eye(3) # as good a set as any other
    result = rotate_vecs_about_axis(np.asarray(1.0),
                                    np.r_[0.0, 1.0, 0.0],
                                    vecs)
    assert result.shape == vecs.shape


def test_correct_single_angle_multiple_axis():
    vecs = np.eye(3)
    result = rotate_vecs_about_axis(np.asarray(1.0),
                                    vecs,
                                    vecs)
    assert result.shape == vecs.shape


def test_correct_multiple_angle_single_axis():
    vecs = np.eye(3)
    result = rotate_vecs_about_axis(np.r_[0.0, 0.5, 1.0],
                                    np.r_[0.0, 1.0, 0.0],
                                    vecs)
    assert result.shape == vecs.shape

def test_correct_multiple_angle_axis():
    vecs = np.eye(3)
    result = rotate_vecs_about_axis(np.r_[0.0, 0.5, 1.0],
                                    vecs,
                                    vecs)
    assert result.shape == vecs.shape
        

def test_incorrect_angle():
    # incorrect: None angle not allowed
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(None,
                               np.eye(3),
                               np.eye(3))

    # incorrect: direct scalar not allowed (this may change). Expects ndarray that may have 0 dims
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(1.0,
                               np.eye(3),
                               np.eye(3))

    # incorrect: too many angle dimensions
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.eye(3),
                               np.eye(3),
                               np.eye(3))

    # incorrect: dimension mismatch
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.r_[0, 0.5],
                               np.eye(3),
                               np.eye(3))

    # incorrect: not double
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.ones((3,), dtype=int),
                               np.eye(3),
                               np.eye(3))


def test_incorrect_axis():
    # incorrect: None axis not allowed
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               None,
                               np.eye(3))

    # incorrect: incorrect dimensions
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               np.asarray(1.0),
                               np.eye(3))

        
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               np.ones((3,3,3)),
                               np.eye(3))

    # incorrect: dimension mismatch
    # N mismatch
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               np.ones((4,3)),
                               np.eye(3))

    # incorrect: not 3 vectors
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               np.ones((3,2)),
                               np.eye(3))

    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               np.ones((2,)),
                               np.eye(3))

    # incorrect: not doubles
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               np.ones((3,), dtype=int),
                               np.eye(3))


def test_incorrect_vecs():
    # incorrect: None vecs not allowed
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               np.r_[0.0, 0.0, 1.0],
                               None)

    # incorrect: incorrect dimensions
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               np.r_[0.0, 0.0, 1.0],
                               np.r_[1.0, 1.0, 1.0])

        
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               np.r_[0.0, 0.0, 1.0],
                               np.ones((3, 3, 3)))

    # incorrect: not 3 vectors
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               np.r_[0.0, 0.0, 1.0],
                               np.ones((3,2)))

    # incorrect: not doubles
    with pytest.raises(ValueError):
        rotate_vecs_about_axis(np.asarray(1.0),
                               np.r_[0.0, 0.0, 1.0],
                               np.ones((3,), dtype=int))
