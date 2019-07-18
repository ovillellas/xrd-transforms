# tests for xy_to_gvec

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_new_capi
from common import xf_numba

all_impls = pytest.mark.parametrize('xy_to_gvec_impl, module_name', 
                                    [(xf_numpy.xy_to_gvec, 'numpy'),
                                     (xf_capi.xy_to_gvec, 'capi'),
                                     (xf_new_capi.xy_to_gvec, 'new_capi'),
                                     #(xf_numba.angles_to_gvec, 'numba'),
                                     (xf.xy_to_gvec, 'default')]
                                )


@all_impls
def test_sample1(xy_to_gvec_impl, module_name):
    pass

@all_impls
def test_sample2(xy_to_gvec_impl, module_name):
    pass
