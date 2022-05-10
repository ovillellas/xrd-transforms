# tests for gvec_to_rays

from __future__ import absolute_import

from collections import namedtuple

import pytest

import numpy as np
from numpy.testing import assert_allclose

from common import function_implementations, convert_axis_angle_to_rmat


all_impls = pytest.mark.parametrize('gvec_to_rays_impl, module_name',
                                    function_implementations('gvec_to_rays'))


# gvec_to_rays uses a reduced argument list wrt the previous gvec_to_xy. This is
# due to being 'detector independent'.
Experiment = namedtuple('Experiment', ['rtol', 'gvec_c', 'rmat_s', 'rmat_c',
                                       'tvec_s', 'tvec_c', 'result'])


@pytest.fixture(scope='module')
def experiment():
    yield Experiment(
        rtol=1e-6,
        gvec_c=np.array([[ 0.57735027,  0.57735028,  0.57735027],
                         [ 0.57735027, -0.57735027,  0.57735028],
                         [ 0.57735027, -0.57735028, -0.57735026],
                         [ 0.57735028,  0.57735027, -0.57735027]]),
        rmat_s=np.array([[[ 0.77029942,  0.        ,  0.63768237],
                          [ 0.        ,  1.        , -0.        ],
                          [-0.63768237,  0.        ,  0.77029942]],

                         [[ 0.97986016,  0.        , -0.19968493],
                          [-0.        ,  1.        , -0.        ],
                          [ 0.19968493,  0.        ,  0.97986016]],

                         [[ 0.30523954,  0.        , -0.9522756 ],
                          [-0.        ,  1.        , -0.        ],
                          [ 0.9522756 ,  0.        ,  0.30523954]],

                         [[ 0.73506994,  0.        , -0.67799129],
                          [-0.        ,  1.        , -0.        ],
                          [ 0.67799129,  0.        ,  0.73506994]]]),
        rmat_c=np.array([[ 0.91734473, -0.08166131,  0.38962815],
                         [ 0.31547749,  0.74606417, -0.58639766],
                         [-0.24280159,  0.66084771,  0.71016033]]),
        tvec_s=np.array([0., 0., 0.]),
        tvec_c=np.array([[-0.25, -0.25, -0.25],
                         [-0.25, -0.24, -0.25]]),
    )

##############################################################################
# Test arguments and results dimensions
##############################################################################
@all_impls
def test_result_dimensions(gvec_to_rays_impl, module_name):
    # only vector gvec
    v, p = gvec_to_rays_impl(np.zeros((2,3)), np.zeros((3,3)), np.zeros((3,3)),
                             np.zeros((3,)), np.zeros((3,)))
    assert v.shape == (2,3) # vector per reflection
    assert p.shape == (3,) # a single origin point

    # gvec and rmat_s are vectors (same outer dimension)
    v, p = gvec_to_rays_impl(np.zeros((2,3)), np.zeros((2,3,3)), np.zeros((3,3)),
                             np.zeros((3,)), np.zeros((3,)))
    assert v.shape == (2,3) # vector per reflection
    assert p.shape == (2,3) # origin point per gvec (as rmat_s varies)

    # gvec and tvec_c vectored, single rmat_s
    v, p = gvec_to_rays_impl(np.zeros((2,3)), np.zeros((3,3)), np.zeros((3,3)),
                             np.zeros((3,)), np.zeros((4,3)))
    assert v.shape == (2,3) # vector per reflection
    assert p.shape == (4,3) # origin point per voxel

    # gvec and tvec_c vectored, rmat_s per gvec
    v, p = gvec_to_rays_impl(np.zeros((2,3)), np.zeros((2,3,3)), np.zeros((3,3)),
                             np.zeros((3,)), np.zeros((4,3)))
    assert v.shape == (2,3) # vector per reflection
    assert p.shape == (4,2,3) # origin point per voxel
