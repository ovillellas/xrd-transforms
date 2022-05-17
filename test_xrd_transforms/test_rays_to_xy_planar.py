# tests for rays_to_xy_planar

from __future__ import absolute_import

from collections import namedtuple

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_raises

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


@all_impls
def test_args_vectors(rays_to_xy_planar_impl, module_name):
    # note: vectors is the first arg
    checker = dimensional_checker(rays_to_xy_planar_impl)

    # vectors inner dimension must be 3
    with assert_raises(ValueError):
        checker(((2,), (3,), (3,3), (3,)), origin_per_vector=False)

    with assert_raises(ValueError):
        checker(((2,2), (3,), (3,3), (3,)), origin_per_vector=False)

    # too few dimensions
    with assert_raises(ValueError):
        checker((tuple(), (3,), (3,3), (3,)), origin_per_vector=False)

    # too many dimensions
    with assert_raises(ValueError):
        checker(((2,2,3), (3,), (3,3), (3,)), origin_per_vector=False)

@all_impls
def test_args_origins(rays_to_xy_planar_impl, module_name):
    # note: origins is the second arg
    checker = dimensional_checker(rays_to_xy_planar_impl)

    # origins inner dimension must be 3
    with assert_raises(ValueError):
        checker(((3,), (2,), (3,3), (3,)), origin_per_vector=False)

    with assert_raises(ValueError):
        checker(((3,), (4,2), (3,3), (3,)), origin_per_vector=False) # M=4, N=-

    with assert_raises(ValueError):
        checker(((2,3), (2,2), (3,3), (3,)), origin_per_vector=True) # M=-, N=2

    with assert_raises(ValueError):
        checker(((2,3), (4,2,2,), (3,3), (3,)), origin_per_vector=True) # M=4, N=2

    # too few dimensions
    with assert_raises(ValueError):
        checker(((3,), tuple(), (3,3), (3,)), origin_per_vector=False)

    with assert_raises(ValueError):
        checker(((2,3), (3,), (3,3), (3,)), origin_per_vector=True)

    # too many dimensions
    with assert_raises(ValueError):
        checker(((3,), (5,4,3), (3,3), (3,)), origin_per_vector=False)

    with assert_raises(ValueError):
        checker(((2,3), (5,4,2,3), (3,3), (3,)), origin_per_vector=True)

    # origins-vectors dimension mismatch
    with assert_raises(ValueError):
        checker(((2, 3), (4, 3, 3), (3,3), (3,)), origin_per_vector=True)

    with assert_raises(ValueError):
        checker(((2,3), (2, 4, 3), (3,3), (3,)), origin_per_vector=True)


@all_impls
def test_args_rmat_d(rays_to_xy_planar_impl, module_name):
    # note: rmat_d is the third arg
    checker = dimensional_checker(rays_to_xy_planar_impl)

    # rmat_d must be a (3,3) matrix
    with assert_raises(ValueError):
        checker(((3,), (3,), (2,2), (3,)), origin_per_vector=False)

    # too few dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,), (3,), (3,)), origin_per_vector=False)

    # too many dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,), (5,3,3), (3,)), origin_per_vector=False)


@all_impls
def test_args_tvec_d(rays_to_xy_planar_impl, module_name):
    # note: tvec_d is the fourth arg
    checker = dimensional_checker(rays_to_xy_planar_impl)

    # tvec_d must be a 3-vector
    with assert_raises(ValueError):
        checker(((3,), (3,), (3,3), (2,)), origin_per_vector=False)

    # too few dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,), (3,3), tuple()), origin_per_vector=False)

    # too many dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,), (3,3), (3,3)), origin_per_vector=False)
