# tests for rotate_vecs_about_axis

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_new_capi
from common import xf_numba


all_impls = pytest.mark.parametrize('rotate_vecs_about_axis_impl, module_name', 
                                    [(xf_numpy.rotate_vecs_about_axis, 'numpy'),
                                     (xf_capi.rotate_vecs_about_axis, 'capi'),
                                     (xf_new_capi.rotate_vecs_about_axis, 'new_capi'),
                                     #(xf_numba.angles_to_gvec, 'numba'),
                                     (xf.rotate_vecs_about_axis, 'default')]
                                )


@all_impls
def test_sample1(rotate_vecs_about_axis_impl, module_name):
    pass

@all_impls
def test_sample2(rotate_vecs_about_axis_impl, module_name):
    pass
