# tests for capi implementation of quat_distance.
#
# these tests are more about making sure the function generates the
# right exceptions when incorrect arguments are used

from __future__ import absolute_import

from collections import namedtuple
import pytest
import numpy as np
from xrd_transforms._new_transforms_capi import quat_distance

    
def test_correct_quat_distance():
    q1 = np.r_[1.0, 0.0, 0.0, 0.0]
    q2 = np.r_[0.0, 1.0, 0.0, 0.0]
    qsym = np.vstack([q1, q2])
    result = quat_distance(q1, q2, qsym)
    assert type(result) is float


def test_incorrect_q1():
    q1 = np.r_[1.0, 0.0, 0.0, 0.0]
    q2 = np.r_[0.0, 1.0, 0.0, 0.0]
    qsym = np.vstack([q1, q2])

    # incorrect: None not supported
    with pytest.raises(ValueError):
        quat_distance(None, q2, qsym)
    
    # incorrect dimensions (not a quaternion)
    with pytest.raises(ValueError):
        quat_distance(np.r_[0.0, 1.0, 0.0], q2, qsym)

    # incorrect: multiple dimensions in q1 not supported
    with pytest.raises(ValueError):
        quat_distance(qsym, q2, qsym)

        
def test_incorrect_q2():
    q1 = np.r_[1.0, 0.0, 0.0, 0.0]
    q2 = np.r_[0.0, 1.0, 0.0, 0.0]
    qsym = np.vstack([q1, q2])

    # incorrect: None not supported
    with pytest.raises(ValueError):
        quat_distance(q1, None, qsym)
    
    # incorrect dimensions (not a quaternion)
    with pytest.raises(ValueError):
        quat_distance(q1, np.r_[0.0, 1.0, 0.0], qsym)

    # incorrect: multiple dimensions in q1 not supported
    with pytest.raises(ValueError):
        quat_distance(q1, qsym, qsym)

        
def test_incorrect_qsym():
    q1 = np.r_[1.0, 0.0, 0.0, 0.0]
    q2 = np.r_[0.0, 1.0, 0.0, 0.0]
    qsym = np.vstack([q1, q2])

    # incorrect: None not supported
    with pytest.raises(ValueError):
        quat_distance(q1, q2, None)
    
    # incorrect dimensions: single quaternion
    with pytest.raises(ValueError):
        quat_distance(q1, q2, qsym[0])

    # incorrect dimensions: not quaternions
    with pytest.raises(ValueError):
        quat_distance(q1, q2, qsym[:,0:3])

    # incorrect: too many dimensions
    with pytest.raises(ValueError):
        quat_distance(q1, qsym, qsym[np.newaxis, :, :])


