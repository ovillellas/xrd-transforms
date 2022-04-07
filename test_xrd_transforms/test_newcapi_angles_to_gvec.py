# tests for capi implementation of angles_to_gvec
#
# These tests are about the argument checking of the newcapi
# C-module implementation, rather than the actual code that
# is tested at the API level for all implementations.
#
# Note that currently the CAPI version expects all arguments,
# being the wrapper function who handles the defaults by
# using the right default constant object for the argument.
# angs dimensionality is also handled by the wrapper.

from __future__ import absolute_import

from collections import namedtuple
import pytest
import numpy as np
from xrd_transforms._new_transforms_capi import anglesToGVec as angles_to_gvec

Experiment = namedtuple('Experiment', ['angs', 'beam_vec', 'eta_vec', 'chi', 'rmat_c'])

@pytest.fixture(scope='module')
def experiment():
    '''just some simple arguments for anglesToGVec that happen to be right
    different test will just change some of the to an incorrect one to check
    the proper error is raised
    '''
    yield Experiment(angs=np.array([[0.5, 0.4, .03],
                                    [0.6, 0.1, -0.7]]),
                     beam_vec=np.r_[0.0, 0.0, 1.0],
                     eta_vec=np.r_[0.0, 1.0, 0.0],
                     chi=0.5,
                     rmat_c=np.eye(3))

    
def test_correct(experiment):
    result = angles_to_gvec(experiment.angs,
                            experiment.beam_vec,
                            experiment.eta_vec,
                            experiment.chi,
                            experiment.rmat_c)
    assert experiment.angs.shape == result.shape

    
def test_incorrect_single_ang(experiment):
    # This is an artifact of implementation. Right now
    # this is not supported, but at some point it could be supported
    # so that the wrapper can be lighter. Remember to update
    # comments if this changes!
    with pytest.raises(ValueError):
        result = angles_to_gvec(experiment.angs[0],
                                experiment.beam_vec,
                                experiment.eta_vec,
                                experiment.chi,
                                experiment.rmat_c)

def test_incorrect_beam_none(experiment):
    with pytest.raises(ValueError):
        result = angles_to_gvec(experiment.angs,
                                None,
                                experiment.eta_vec,
                                experiment.chi,
                                experiment.rmat_c)


def test_incorrect_beam_short(experiment):
    with pytest.raises(ValueError):
        result = angles_to_gvec(experiment.angs,
                                np.r_[0.0, 1.0],
                                experiment.eta_vec,
                                experiment.chi,
                                experiment.rmat_c)


def test_incorrect_eta_none(experiment):
    # This may change in the future if C code handles it, making
    # the wrapper lighter-weight.
    with pytest.raises(ValueError):
        result = angles_to_gvec(experiment.angs,
                                experiment.beam_vec,
                                None,
                                experiment.chi,
                                experiment.rmat_c)

        
def test_incorrect_eta_short(experiment):
    with pytest.raises(ValueError):
        result = angles_to_gvec(experiment.angs,
                                experiment.beam_vec,
                                np.r_[0.0, 1.0],
                                experiment.chi,
                                experiment.rmat_c)


def test_incorrect_chi_none(experiment):
    with pytest.raises(TypeError):
        result = angles_to_gvec(experiment.angs,
                                experiment.beam_vec,
                                experiment.eta_vec,
                                None,
                                experiment.rmat_c)


def test_incorrect_rmat_c_none(experiment):
    with pytest.raises(ValueError):
        result = angles_to_gvec(experiment.angs,
                                experiment.beam_vec,
                                experiment.eta_vec,
                                experiment.chi,
                                None)

def test_incorrect_rmat_c_wrong_ndim(experiment):
    with pytest.raises(ValueError):
        result = angles_to_gvec(experiment.angs,
                                experiment.beam_vec,
                                experiment.eta_vec,
                                experiment.chi,
                                np.r_[0.0, 0.0, 1.0])

        
def test_incorrect_rmat_c_wrong_dims(experiment):
    with pytest.raises(ValueError):
        result = angles_to_gvec(experiment.angs,
                                experiment.beam_vec,
                                experiment.eta_vec,
                                experiment.chi,
                                np.eye(4))

