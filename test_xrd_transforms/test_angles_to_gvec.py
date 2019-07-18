# tests for angles_to_gvec

from __future__ import absolute_import

import pytest
import numpy as np
import numpy.testing as np_testing

from common import xf
from common import xf_numpy
from common import xf_capi
from common import xf_new_capi
from common import xf_numba


all_impls = pytest.mark.parametrize('angles_to_gvec_impl, module_name', 
                                    [(xf_numpy.angles_to_gvec, 'numpy'),
                                     (xf_capi.angles_to_gvec, 'capi'),
                                     (xf_new_capi.angles_to_gvec, 'new_capi'),
                                     (xf_numba.angles_to_gvec, 'numba'),
                                     (xf.angles_to_gvec, 'default')]
                                )

@all_impls
def test_simple(angles_to_gvec_impl, module_name):
    bHat = np.r_[0.0, 0.0, -1.0]
    eHat = np.r_[1.0, 0.0, 0.0]
    angs = np.array([np.pi, 0.0], dtype= np.double)
    expected = np.r_[0.0, 0.0, 1.0]

    # single entry codepath
    res = angles_to_gvec_impl(angs, bHat, eHat)
    np_testing.assert_almost_equal(res, expected)

    # vector codepath (should return dimensions accordingly)
    res = angles_to_gvec_impl(np.atleast_2d(angs), bHat, eHat)
    np_testing.assert_almost_equal(res, np.atleast_2d(expected))

