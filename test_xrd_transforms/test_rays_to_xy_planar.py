# tests for rays_to_xy_planar

from __future__ import absolute_import

from collections import namedtuple

import pytest

import numpy as np
from numpy.testing import assert_allclose

from common import function_implementations


all_impls = pytest.mark.parametrize('rays_to_xy_planar_impl, module_name',
                                    function_implementations('rays_to_xy_planar'))


##############################################################################
# Test arguments and result dimensions.
##############################################################################
def dimensional_checker(fn):
    '''helper function that will pass to the function fn as many np.zeros arguments
    as shapes are provided as inputs. It will return the shape of the results.

    This can help the test the argument dimension checks by just providing the
    wanted dimensions, with filler data. Some tests are there just to check the
    supported shapes.

    This is slightly different in the way it handles the results compared to
    similar code in test_gvec_to_rays.

    '''
    def _dim_checker(dims, **kw):
        results = fn(*map(np.zeros, dims), **kw)
        return results.shape
    return _dim_checker


@all_impls
def test_result_dimensions(rays_to_xy_planar_impl, module_name):
    checker = dimensional_checker(rays_to_xy_planar_impl)

    # (1,1) batch of rays. Origin per vector should be indifferent in this case
    r = checker(((3,), (3,), (3,3), (3,)), origin_per_vector=False)
    assert r==(2,)
    
    r = checker(((3,), (3,), (3,3), (3,)), origin_per_vector=True)
    assert r==(2,)

    # (M,1) batch of rays (different origins, same vector).
    # Origin per vector still indifferent
    r = checker(((3,), (4,3), (3,3), (3,)), origin_per_vector=False)
    assert r==(4,2)
    r = checker(((3,), (4,3), (3,3), (3,)), origin_per_vector=True)
    assert r==(4,2)

    # (1,N) batch of rays (same origin, different vectors).
    # Origin per vector becomes relevant.
    r = checker(((2,3), (3,), (3,3), (3,)), origin_per_vector=False)
    assert r==(2,2)
    r = checker(((2,3), (2,3), (3,3), (3,)), origin_per_vector=True)
    assert r==(2,2)

    # (M,N) batch of rays (different origins, different vectors)
    # Origin per vector also relevant in this case
    r = checker(((2,3), (4,3), (3,3), (3,)), origin_per_vector=False)
    assert r==(4,2,2)
    r = checker(((2,3), (4,2,3), (3,3), (3,)), origin_per_vector=True)
    assert r==(4,2,2)
    
