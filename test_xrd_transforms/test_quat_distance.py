# tests for quat_distance

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_new_capi
from common import xf_numba


all_impls = pytest.mark.parametrize('quat_distance_impl, module_name', 
                                    [(xf_numpy.quat_distance, 'numpy'),
                                     (xf_capi.quat_distance, 'capi'),
                                     (xf_new_capi.quat_distance, 'new_capi'),
                                     #(xf_numba.angles_to_gvec, 'numba'),
                                     (xf.quat_distance, 'default')]
                                )


@all_impls
def test_sample1(quat_distance_impl, module_name):
    pass

@all_impls
def test_sample2(quat_distance_impl, module_name):
    pass

