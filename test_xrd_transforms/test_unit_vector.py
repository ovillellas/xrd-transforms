# tests for unit_vector

from __future__ import absolute_import

import pytest

import numpy as np
from numpy.testing import assert_allclose

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_new_capi
from common import xf_numba

all_impls = pytest.mark.parametrize('unit_vector_impl, module_name', 
                                    [(xf_numpy.unit_vector, 'numpy'),
                                     (xf_capi.unit_vector, 'capi'),
                                     (xf_new_capi.unit_vector, 'new_capi'),
                                     (xf_numba.unit_vector, 'numba'),
                                     (xf.unit_vector, 'default')]
)


def _get_random_vectors_array():
    # return a (n,3) array with some vectors and a (n) array with the expected
    # result norms.
    arr = np.array([[42.0,  0.0,  0.0, 1.0],
                    [12.0, 12.0, 12.0, 1.0],
                    [ 0.0,  0.0,  0.0, 0.0],
                    [ 0.7, -0.7,  0.0, 1.0],
                    [-0.0, -0.0, -0.0, 0.0]])
    return arr[:,0:3], arr[:,3]


# ------------------------------------------------------------------------------

# trivial tests

@all_impls
def test_trivial(unit_vector_impl, module_name):
    # all vectors in eye(3) are already unit vectors
    iden = np.eye(3)

    # check a vector at a time
    assert_allclose(unit_vector_impl(iden[0]), iden[0])
    assert_allclose(unit_vector_impl(iden[1]), iden[1])
    assert_allclose(unit_vector_impl(iden[2]), iden[2])

    # use the array version
    assert_allclose(unit_vector_impl(iden), iden)


@all_impls
def test_zero(unit_vector_impl, module_name):
    # When a zero vector is given, a potential "division by zero" happens.
    # in this library, instead of trying to normalize a zero-norm vector
    # (which would trigger the division by zero), the original vector is
    # returned.

    # check vector
    zero_vec = np.zeros((3,))
    assert_allclose(unit_vector_impl(zero_vec), zero_vec)

    # check array
    zero_arr = np.zeros((3,3))
    assert_allclose(unit_vector_impl(zero_arr), zero_arr)

    # check mixed array
    mix_arr = np.eye(3)
    mix_arr[1,:] = 0.0
    assert_allclose(unit_vector_impl(mix_arr), mix_arr)


@all_impls
def test_random_vectors(unit_vector_impl, module_name):
    # test for some random vectors. The test just checks that the norm of the
    # the resulting vector is as expected.

    vecs, expected_norm = _get_random_vectors_array()

    # element by element
    for i in range(len(vecs)):
        result = unit_vector_impl(vecs[i])
        assert_allclose(np.linalg.norm(result), expected_norm[i])

    # all in a row
    result = unit_vector_impl(vecs)
    assert_allclose(np.linalg.norm(result, axis=1), expected_norm)


# ------------------------------------------------------------------------------

# check input ordering

@all_impls
def test_strided_inputs(unit_vector_impl, module_name):
    vecs, expected_norm = _get_random_vectors_array()

    vecs_f = np.asfortranarray(vecs)

    # element by element
    for i in range(len(vecs_f)):
        result = unit_vector_impl(vecs_f[i])
        assert_allclose(np.linalg.norm(result), expected_norm[i])

    # all in a row
    result = unit_vector_impl(vecs_f)
    assert_allclose(np.linalg.norm(result, axis=1), expected_norm)

