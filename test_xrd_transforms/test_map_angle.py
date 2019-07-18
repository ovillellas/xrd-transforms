# tests for map_angle

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_numba

all_impls = pytest.mark.parametrize('map_angle_impl, module_name', 
                                    [(xf_numpy.map_angle, 'numpy'),
                                     #(xf_capi.map_angle, 'capi'),
                                     #(xf_numba.angles_to_gvec, 'numba'),
                                     (xf.map_angle, 'default')]
                                )


@all_impls
def test_sample1(map_angle_impl, module_name):
    pass

@all_impls
def test_sample2(map_angle_impl, module_name):
    pass
