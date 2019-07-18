# tests for row_norm

from __future__ import absolute_import

import pytest
import numpy as np
from numpy.testing import assert_allclose

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_new_capi
from common import xf_numba


all_impls = pytest.mark.parametrize('row_norm_impl, module_name', 
                                    [(xf_numpy.row_norm, 'numpy'),
                                     #(xf_capi.row_norm, 'capi'),
                                     #(xf_new_capi.row_norm, 'new_capi'),
                                     (xf_numba.row_norm, 'numba'),
                                     (xf.row_norm, 'default')]
                                )


def _get_random_vectors_array():
    # return a (n,3) array with some vectors and a (n) array with the expected
    # result norms.
    arr = np.array([[42.0,  0.0,  0.0],
                    [12.0, 12.0, 12.0],
                    [ 0.0,  0.0,  0.0],
                    [ 0.7, -0.7,  0.0],
                    [-0.0, -0.0, -0.0]])
    return arr


@all_impls
def test_random_vectors(row_norm_impl, module_name):
    # checking against numpy.linalg.norm
    vecs = _get_random_vectors_array()

    # element by element
    for i in range(len(vecs)):
        result = row_norm_impl(vecs[i])
        expected = np.linalg.norm(vecs[i])
        assert type(result) == type(expected)
        assert result.dtype == expected.dtype
        assert_allclose(result, expected)

    # all in a row
    result = row_norm_impl(vecs)
    expected = np.linalg.norm(vecs, axis=1)
    assert type(result) == type(expected)
    assert result.dtype == expected.dtype
    
    assert_allclose(result, expected)


@all_impls
def test_too_many_dimensions(row_norm_impl, module_name):
    # our norm should fail on 3 dimensional arrays using a ValueError
    test_vec = np.arange(16., dtype=np.double).reshape((4,2,2))
    with pytest.raises(ValueError):
        row_norm_impl(test_vec)

    
