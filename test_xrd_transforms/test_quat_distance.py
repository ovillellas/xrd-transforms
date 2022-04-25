# tests for quat_distance

from __future__ import absolute_import

import pytest
import numpy as np
from numpy.testing import assert_allclose

from common import function_implementations


all_impls = pytest.mark.parametrize('quat_distance_impl, module_name', 
                                    function_implementations('quat_distance'))


@all_impls
def test_simple(quat_distance_impl, module_name):
    q1 = np.r_[1.0, 0.0, 0.0, 0.0] #identity
    q2 = q1
    qsym = np.array([[1.0, 0.0, 0.0, 0.0],
                     [-1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

    result = quat_distance_impl(q1, q2, qsym.T)

    # distance to the same quaternion should always be 0.0
    assert_allclose(result, 0.0)


@all_impls
def test_distance(quat_distance_impl, module_name):
    q1 = np.r_[1.0, 0.0, 0.0, 0.0] #identity
    q2 = np.r_[0.0, 0.0, 0.0, -1.0]
    qsym = np.array([[1.0, 0.0, 0.0, 0.0],
                     [-1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

    # applying the last symmetry should result in 0.0 distance
    result = quat_distance_impl(q1, q2, qsym.T)
    assert_allclose(result, 0.0)

    
    result = quat_distance_impl(q1, q2, qsym.T[:,:-1])
    assert_allclose(result, np.pi)

