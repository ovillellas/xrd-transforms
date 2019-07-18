# tests for row_norm

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_numba


all_impls = pytest.mark.parametrize('row_norm_impl, module_name', 
                                    [(xf_numpy.row_norm, 'numpy'),
                                     #(xf_capi.row_norm, 'capi'),
                                     (xf_numba.row_norm, 'numba'),
                                     (xf.row_norm, 'default')]
                                )


@all_impls
def test_sample1(row_norm_impl, module_name):
    pass

@all_impls
def test_sample2(row_norm_impl, module_name):
    pass
