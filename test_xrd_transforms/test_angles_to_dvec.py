# tests for angles_to_dvec

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_new_capi
from common import xf_numba


all_impls = pytest.mark.parametrize('angles_to_dvec_impl, module_name', 
                                    [(xf_numpy.angles_to_dvec, 'numpy'),
                                     (xf_capi.angles_to_dvec, 'capi'),
                                     (xf_new_capi.angles_to_dvec, 'new_capi'),
                                     (xf_numba.angles_to_dvec, 'numba'),
                                     (xf.angles_to_dvec, 'default')]
                                )


@all_impls
def test_sample1(angles_to_dvec_impl, module_name):
    pass

@all_impls
def test_sample2(angles_to_dvec_impl, module_name):
    pass
