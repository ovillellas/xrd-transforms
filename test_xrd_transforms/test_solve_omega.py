# tests for solve_omega

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_new_capi
from common import xf_numba

all_impls = pytest.mark.parametrize('solve_omega_impl, module_name', 
                                    [(xf_numpy.solve_omega, 'numpy'),
                                     #(xf_capi.solve_omega, 'capi'),
                                     #(xf_new_capi.solve_omega, 'new_capi'),
                                     #(xf_numba.solve_omega, 'numba'),
                                     (xf.solve_omega, 'default')]
                                )


@all_impls
def test_sample1(solve_omega_impl, module_name):
    pass

@all_impls
def test_sample2(solve_omega_impl, module_name):
    pass
