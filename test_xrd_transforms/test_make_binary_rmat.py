# tests for make_binary_rmat

from __future__ import absolute_import

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_new_capi
from common import xf_numba

import pytest

all_impls = pytest.mark.parametrize('make_binary_rmat_impl, module_name', 
                                    [(xf_numpy.make_binary_rmat, 'numpy'),
                                     (xf_capi.make_binary_rmat, 'capi'),
                                     (xf_new_capi.make_binary_rmat, 'new_capi'),
                                     #(xf_numba.angles_to_gvec, 'numba'),
                                     (xf.make_binary_rmat, 'default')]
                                )


@all_impls
def test_sample1(make_binary_rmat_impl, module_name):
    pass

@all_impls
def test_sample2(make_binary_rmat_impl, module_name):
    pass
