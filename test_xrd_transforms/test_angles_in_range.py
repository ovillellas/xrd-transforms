# tests for angles_in_range

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_numba

all_impls = pytest.mark.parametrize('angles_in_range_impl, module_name', 
                                    [(xf_numpy.angles_in_range, 'numpy'),
                                     #(xf_capi.angles_in_range, 'capi'),
                                     #(xf_numba.angles_to_gvec, 'numba'),
                                     (xf.angles_in_range, 'default')]
                                )


@all_impls
def test_sample1(angles_in_range_impl, module_name):
    pass

@all_impls
def test_sample2(angles_in_range_impl, module_name):
    pass
