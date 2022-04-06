# tests for capi implementation of gvec_to_xy.
#
# these tests are more about making sure the function generates the
# right exceptions when incorrect arguments are used

from __future__ import absolute_import

from collections import namedtuple
import pytest
import numpy as np
from xrd_transforms._new_transforms_capi import gvecToDetectorXY as gvec_to_xy
from xrd_transforms._new_transforms_capi import gvecToDetectorXYArray as gvec_to_xy_array

Experiment = namedtuple('Experiment', ['rtol', 'gvec_c', 'rmat_d', 'rmat_s', 'rmat_c',
                                       'tvec_d', 'tvec_s', 'tvec_c', 'result'])

@pytest.fixture(scope='module')
def experiment():
    '''Note this fixture is data is actually extracted from some test runs. They
    are assumed to be correct, but no actual hard checking has been done.
    This is just a subset. It should be enough to exercise the gvec_to_xy
    function with different vector/scalar arrangements'''
    yield Experiment(
        rtol=1e-6,
        gvec_c=np.array([[ 0.57735027,  0.57735028,  0.57735027],
                         [ 0.57735027, -0.57735027,  0.57735028],
                         [ 0.57735027, -0.57735028, -0.57735026],
                         [ 0.57735028,  0.57735027, -0.57735027]]),
        rmat_d=np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]]),
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
        tvec_d=np.array([ 0. ,  1.5, -5. ]),
        tvec_s=np.array([0., 0., 0.]),
        tvec_c=np.array([-0.25, -0.25, -0.25]),
        result=np.array([[ 0.13349048, -1.61131393],
                         [ 0.19186549, -2.03119741],
                         [ 0.63614123, -1.70709656],
                         [ 0.12934705, -1.29999638]])
    )

def test_incorrect_single(experiment):
    # gvec is not supported to be just one dimensional
    with pytest.raises(ValueError):
        result = gvec_to_xy(experiment.gvec_c[0],
                            experiment.rmat_d,
                            experiment.rmat_s[0],
                            experiment.rmat_c,
                            experiment.tvec_d,
                            experiment.tvec_s,
                            experiment.tvec_c,
                            np.r_[0.0, 0.0, -1.0])


def test_correct_gvec_to_xy(experiment):
    result = gvec_to_xy(experiment.gvec_c,
                        experiment.rmat_d,
                        experiment.rmat_s[0],
                        experiment.rmat_c,
                        experiment.tvec_d,
                        experiment.tvec_s,
                        experiment.tvec_c,
                        np.r_[0.0, 0.0, -1.0])

    with pytest.raises(ValueError):
        # multiple rmat_s fails
        result = gvec_to_xy(experiment.gvec_c,
                            experiment.rmat_d,
                            experiment.rmat_s,
                            experiment.rmat_c,
                            experiment.tvec_d,
                            experiment.tvec_s,
                            experiment.tvec_c,
                            np.r_[0.0, 0.0, -1.0])

        
def test_correct_multiple_rmat_s(experiment):
    with pytest.raises(ValueError):
        # array version needs many rmat_s
        result = gvec_to_xy_array(experiment.gvec_c,
                                  experiment.rmat_d,
                                  experiment.rmat_s[0],
                                  experiment.rmat_c,
                                  experiment.tvec_d,
                                  experiment.tvec_s,
                                  experiment.tvec_c,
                                  np.r_[0.0, 0.0, -1.0])
        
    result = gvec_to_xy_array(experiment.gvec_c,
                              experiment.rmat_d,
                              experiment.rmat_s,
                              experiment.rmat_c,
                              experiment.tvec_d,
                              experiment.tvec_s,
                              experiment.tvec_c,
                              np.r_[0.0, 0.0, -1.0])

    
def test_correct_gvec_to_xy_none_beam(experiment):
    result = gvec_to_xy(experiment.gvec_c,
                        experiment.rmat_d,
                        experiment.rmat_s[0],
                        experiment.rmat_c,
                        experiment.tvec_d,
                        experiment.tvec_s,
                        experiment.tvec_c,
                        None)

    result = gvec_to_xy(experiment.gvec_c,
                        experiment.rmat_d,
                        experiment.rmat_s[0],
                        experiment.rmat_c,
                        experiment.tvec_d,
                        experiment.tvec_s,
                        experiment.tvec_c)

    
def test_correct_multiple_rmat_s_none_beam(experiment):
    result = gvec_to_xy_array(experiment.gvec_c,
                              experiment.rmat_d,
                              experiment.rmat_s,
                              experiment.rmat_c,
                              experiment.tvec_d,
                              experiment.tvec_s,
                              experiment.tvec_c,
                              None)

    result = gvec_to_xy_array(experiment.gvec_c,
                              experiment.rmat_d,
                              experiment.rmat_s,
                              experiment.rmat_c,
                              experiment.tvec_d,
                              experiment.tvec_s,
                              experiment.tvec_c)


def test_incorrect_rmat_d(experiment):
    with pytest.raises(ValueError):
        result = gvec_to_xy(experiment.gvec_c,
                            np.eye(4),
                            experiment.rmat_s[0],
                            experiment.rmat_c,
                            experiment.tvec_d,
                            experiment.tvec_s,
                            experiment.tvec_c)
        
    with pytest.raises(ValueError):
        result = gvec_to_xy_array(experiment.gvec_c,
                                  np.eye(4),
                                  experiment.rmat_s,
                                  experiment.rmat_c,
                                  experiment.tvec_d,
                                  experiment.tvec_s,
                                  experiment.tvec_c)


def test_incorrect_rmat_s(experiment):
    with pytest.raises(ValueError):
        result = gvec_to_xy(experiment.gvec_c,
                            experiment.rmat_d,
                            np.eye(4),
                            experiment.rmat_c,
                            experiment.tvec_d,
                            experiment.tvec_s,
                            experiment.tvec_c)
        
    with pytest.raises(ValueError):
        result = gvec_to_xy_array(experiment.gvec_c,
                                  experiment.rmat_d,
                                  np.eye(4),
                                  experiment.rmat_c,
                                  experiment.tvec_d,
                                  experiment.tvec_s,
                                  experiment.tvec_c)

def test_incorrect_rmat_c(experiment):
    with pytest.raises(ValueError):
        result = gvec_to_xy(experiment.gvec_c,
                            experiment.rmat_d,
                            experiment.rmat_s[0],
                            np.eye(4),
                            experiment.tvec_d,
                            experiment.tvec_s,
                            experiment.tvec_c)
        
    with pytest.raises(ValueError):
        result = gvec_to_xy_array(experiment.gvec_c,
                                  experiment.rmat_d,
                                  experiment.rmat_s,
                                  np.eye(4),
                                  experiment.tvec_d,
                                  experiment.tvec_s,
                                  experiment.tvec_c)

def test_incorrect_tvec_d(experiment):
    with pytest.raises(ValueError):
        result = gvec_to_xy(experiment.gvec_c,
                            experiment.rmat_d,
                            experiment.rmat_s[0],
                            experiment.rmat_c,
                            np.r_[0.0, 0.0],
                            experiment.tvec_s,
                            experiment.tvec_c)
        
    with pytest.raises(ValueError):
        result = gvec_to_xy_array(experiment.gvec_c,
                                  experiment.rmat_d,
                                  experiment.rmat_s,
                                  experiment.rmat_c,
                                  np.r_[0.0, 0.0],
                                  experiment.tvec_s,
                                  experiment.tvec_c)

def test_incorrect_tvec_s(experiment):
    with pytest.raises(ValueError):
        result = gvec_to_xy(experiment.gvec_c,
                            experiment.rmat_d,
                            experiment.rmat_s[0],
                            experiment.rmat_c,
                            experiment.tvec_d,
                            np.r_[0.0, 0.0],
                            experiment.tvec_c)
        
    with pytest.raises(ValueError):
        result = gvec_to_xy_array(experiment.gvec_c,
                                  experiment.rmat_d,
                                  experiment.rmat_s,
                                  experiment.rmat_c,
                                  experiment.tvec_d,
                                  np.r_[0.0, 0.0],
                                  experiment.tvec_c)

def test_incorrect_tvec_c(experiment):
    with pytest.raises(ValueError):
        result = gvec_to_xy(experiment.gvec_c,
                            experiment.rmat_d,
                            experiment.rmat_s[0],
                            experiment.rmat_c,
                            experiment.tvec_d,
                            experiment.tvec_s,
                            np.r_[0.0, 0.0])
        
    with pytest.raises(ValueError):
        result = gvec_to_xy_array(experiment.gvec_c,
                                  experiment.rmat_d,
                                  experiment.rmat_s,
                                  experiment.rmat_c,
                                  experiment.tvec_d,
                                  experiment.tvec_s,
                                  np.r_[0.0, 0.0])

        
def test_incorrect_beam(experiment):
    with pytest.raises(ValueError):
        result = gvec_to_xy(experiment.gvec_c,
                            experiment.rmat_d,
                            experiment.rmat_s[0],
                            experiment.rmat_c,
                            experiment.tvec_d,
                            experiment.tvec_s,
                            experiment.tvec_c,
                            np.r_[0.0, 0.0])
        
    with pytest.raises(ValueError):
        result = gvec_to_xy_array(experiment.gvec_c,
                                  experiment.rmat_d,
                                  experiment.rmat_s,
                                  experiment.rmat_c,
                                  experiment.tvec_d,
                                  experiment.tvec_s,
                                  experiment.tvec_c,
                                  np.r_[0.0, 0.0])

        
def test_incorrect_beam_dtype(experiment):
    with pytest.raises(ValueError):
        result = gvec_to_xy(experiment.gvec_c,
                            experiment.rmat_d,
                            experiment.rmat_s[0],
                            experiment.rmat_c,
                            experiment.tvec_d,
                            experiment.tvec_s,
                            experiment.tvec_c,
                            np.r_[0, 0, -1]) # int dtype!
        
    with pytest.raises(ValueError):
        result = gvec_to_xy_array(experiment.gvec_c,
                                  experiment.rmat_d,
                                  experiment.rmat_s,
                                  experiment.rmat_c,
                                  experiment.tvec_d,
                                  experiment.tvec_s,
                                  experiment.tvec_c,
                                  np.r_[0, 0, -1]) # int dtype!

