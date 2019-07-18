# tests for validate_angle_ranges

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_numba


all_impls = pytest.mark.parametrize('validate_angle_ranges_impl, module_name', 
                                    [(xf_numpy.validate_angle_ranges, 'numpy'),
                                     (xf_capi.validate_angle_ranges, 'capi'),
                                     #(xf_numba.angles_to_gvec, 'numba'),
                                     (xf.validate_angle_ranges, 'default')]
                                )


@all_impls
def test_sample1(validate_angle_ranges_impl, module_name):
    pass

@all_impls
def test_sample2(validate_angle_ranges_impl, module_name):
    pass
