# tests for angular_difference

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_new_capi
from common import xf_numba

all_impls = pytest.mark.parametrize('angular_difference_impl, module_name', 
                                    [(xf_numpy.angular_difference, 'numpy'),
                                     #(xf_capi.angular_difference, 'capi'),
                                     #(xf_new_capi.angular_difference, 'new_capi'),
                                     #(xf_numba.angles_to_gvec, 'numba'),
                                     (xf.angular_difference, 'default')]
                                )


@all_impls
def test_sample1(angular_difference_impl, module_name):
    pass

@all_impls
def test_sample2(angular_difference_impl, module_name):
    pass
