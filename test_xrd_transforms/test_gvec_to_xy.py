# tests for angles_to_gvec

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_new_capi
from common import xf_numba


all_impls = pytest.mark.parametrize('gvec_to_xy_impl, module_name', 
                                    [(xf_numpy.gvec_to_xy, 'numpy'),
                                     (xf_capi.gvec_to_xy, 'capi'),
                                     (xf_new_capi.gvec_to_xy, 'new_capi'),
                                     #(xf_numba.gvec_to_xy, 'numba'),
                                     (xf.gvec_to_xy, 'default')]
                                )


@all_impls
def test_sample1(gvec_to_xy_impl, module_name):
    pass

@all_impls
def test_sample2(gvec_to_xy_impl, module_name):
    pass
