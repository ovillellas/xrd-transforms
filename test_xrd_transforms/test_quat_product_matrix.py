# tests for quat_product_matrix

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_numba


all_impls = pytest.mark.parametrize('quat_product_matrix_impl, module_name', 
                                    [(xf_numpy.quat_product_matrix, 'numpy'),
                                     #(xf_capi.quat_product_matrix, 'capi'),
                                     #(xf_numba.angles_to_gvec, 'numba'),
                                     (xf.quat_product_matrix, 'default')]
                                )


@all_impls
def test_sample1(quat_product_matrix_impl, module_name):
    pass

@all_impls
def test_sample2(quat_product_matrix_impl, module_name):
    pass

