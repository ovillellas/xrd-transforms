# tests for make_sample_rmat

from __future__ import absolute_import

import pytest

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_numba


all_impls = pytest.mark.parametrize('make_sample_rmat_impl, module_name', 
                                    [(xf_numpy.make_sample_rmat, 'numpy'),
                                     #(xf_capi.make_sample_rmat, 'capi'),
                                     #(xf_numba.angles_to_gvec, 'numba'),
                                     (xf.make_sample_rmat, 'default')]
                                )


@all_impls
def test_sample1(make_sample_rmat_impl, module_name):
    pass

@all_impls
def test_sample2(make_sample_rmat_impl, module_name):
    pass
