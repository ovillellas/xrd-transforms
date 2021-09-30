# tests for xy_to_gvec

from __future__ import absolute_import

import pytest

import numpy as np
from numpy.testing import assert_allclose

from common import function_implementations

all_impls = pytest.mark.parametrize('xy_to_gvec_impl, module_name', 
                                    function_implementations('xy_to_gvec'))


# xy_to_gvec takes parametric coordinates in the DETECTOR plane and calculates
# the associated gvectors.
#
# xy_to_gvec can be seen as the inverse of gvec_to_xy
#
# arguments:
# xy_d: (n,2) array. (x, y) coordinates in the DETECTOR plane.
# rmat_d: (3,3) array. COB matrix from DETECTOR to LAB frame.
# rmat_s: (3,3) array. COB matrix from SAMPLE to LAB frame.
# tvec_d: (3,) array. Translation vector from LAB to DETECTOR frame.
# tvec_s: (3,) array. Translation vector from LAB to SAMPLE frame.
# tvec_c: (3,3) array. COB matrix from BEAM to LAB. Defaults to None, implying identity.
# distortion: distortion class, optional. Default is None.
#
# returns
# array_like
#    (n, 2) ndarray containg the (tth, eta) pairs associated with each (x, y)
# array_like
#    (n, 3) ndarray containg the associated G vector directions in LAB frame.
# array_like, optional
#    if output_ref is True

@all_impls
def test_sample1(xy_to_gvec_impl, module_name):
    pass

@all_impls
def test_sample2(xy_to_gvec_impl, module_name):
    pass
