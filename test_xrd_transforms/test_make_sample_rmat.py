# tests for make_sample_rmat

from __future__ import absolute_import

import pytest
from numpy.testing import assert_allclose as np_assert_allclose

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_numba
from common import xf_cnst

all_impls = pytest.mark.parametrize('make_sample_rmat_impl, module_name', 
                                    [(xf_numpy.make_sample_rmat, 'numpy'),
                                     #(xf_capi.make_sample_rmat, 'capi'),
                                     #(xf_numba.angles_to_gvec, 'numba'),
                                     (xf.make_sample_rmat, 'default')]
                                )


@all_impls
def test_make_sample_rmat_chi0_ome0(make_sample_rmat_impl, module_name):
    # when chi = 0.0 and ome = 0.0 the resulting sample rotation matrix should
    # be the identity
    chi = 0.0
    ome = 0.0
    result = make_sample_rmat_impl(0.0, 0.0)

    np_assert_allclose(xf_cnst.identity_3x3, result)


@all_impls
def test_sample1(make_sample_rmat_impl, module_name):
    pass

@all_impls
def test_sample2(make_sample_rmat_impl, module_name):
    pass
