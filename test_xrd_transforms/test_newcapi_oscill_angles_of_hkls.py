# tests for capi implementation of oscill_angles_of_hkls.py
#
# these tests are more about making sure the function generates the
# right exceptions when incorrect arguments are used

from __future__ import absolute_import

from collections import namedtuple
import pytest
import numpy as np
from xrd_transforms._new_transforms_capi import oscillAnglesOfHKLs as oscill_angles_of_hkls

# Note th
oscill_angles_args = namedtuple('oscill_angles_args', ['hkls', 'chi', 'rmat_c',
                                                       'bmat', 'wavelength',
                                                       'v_inv', 'beam_vec',
                                                       'eta_vec'])


@pytest.fixture(scope='module')
def test_args():
    '''
    This fixture only is about argument types and dimensions.
    There is no need for it to reflect a real problem.
    '''
    N = 12
    yield oscill_angles_args(hkls=np.ones((N,3)),
                             chi=0.1,
                             rmat_c=np.eye(3),
                             bmat=np.eye(3),
                             wavelength=0.001,
                             v_inv=np.ones((6,)),
                             beam_vec = np.r_[0.0, 0.0, -1.0],
                             eta_vec = np.r_[0.0, 1.0, 0.0])

    
def test_correct_oscill_angles_of_hkls(test_args):
    result = oscill_angles_of_hkls(*test_args)
    assert len(result) == 2
    assert result[0].shape == test_args.hkls.shape
    assert result[1].shape == test_args.hkls.shape


def test_incorrect_hkls(test_args):
    # only one dimension hkls
    args = test_args._replace(hkls=np.ones((3,)))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)

    # too many dimension hkls
    args = test_args._replace(hkls=np.ones((3,3,3)))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)

    # incorrect inner dimension
    args = test_args._replace(hkls=np.ones((12,5)))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)

    # incorrect None as hkls
    args = test_args._replace(hkls=None)
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)


def test_incorrect_chi(test_args):
    # incorrect: an array
    args = test_args._replace(chi=np.ones((3,)))
    with pytest.raises(TypeError):
        oscill_angles_of_hkls(*args)

    # incorrect: None
    args = test_args._replace(chi=None)
    with pytest.raises(TypeError):
        oscill_angles_of_hkls(*args)


def test_incorrect_rmat_c(test_args):
    # incorrect: not enough dimensions
    args = test_args._replace(rmat_c=np.ones(3,))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)

    # incorrect: too many dimensions
    args = test_args._replace(rmat_c=np.ones((3,3,3)))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)

    # incorrect: not the right dimensions
    args = test_args._replace(rmat_c=np.eye(4))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)
    
    # incorrect: None
    args = test_args._replace(rmat_c=None)
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)


def test_incorrect_bmat(test_args):
    # incorrect: not enough dimensions
    args = test_args._replace(bmat=np.ones(3,))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)

    # incorrect: too many dimensions
    args = test_args._replace(bmat=np.ones((3,3,3)))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)

    # incorrect: not the right dimensions
    args = test_args._replace(bmat=np.eye(4))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)
    
    # incorrect: None
    args = test_args._replace(bmat=None)
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)


def test_incorrect_wavelength(test_args):
    # incorrect: an array
    args = test_args._replace(wavelength=np.ones((3,)))
    with pytest.raises(TypeError):
        oscill_angles_of_hkls(*args)

    # incorrect: None
    args = test_args._replace(wavelength=None)
    with pytest.raises(TypeError):
        oscill_angles_of_hkls(*args)


def test_incorrect_v_inv(test_args):
    # incorrect: too many dimensions
    args = test_args._replace(v_inv=np.ones((3,6)))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)

    # incorrect: not the right dimensions
    args = test_args._replace(v_inv=np.ones(4))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)
    
    # incorrect: None
    args = test_args._replace(v_inv=None)
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)


def test_incorrect_beam_vec(test_args):
    # incorrect: too many dimensions
    args = test_args._replace(beam_vec=np.ones((3,3)))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)

    # incorrect: not the right dimensions
    args = test_args._replace(beam_vec=np.ones(4))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)
    
    # incorrect: None
    args = test_args._replace(beam_vec=None)
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)


def test_incorrect_eta_vec(test_args):
    # incorrect: too many dimensions
    args = test_args._replace(eta_vec=np.ones((3,3)))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)

    # incorrect: not the right dimensions
    args = test_args._replace(eta_vec=np.ones(4))
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)
    
    # incorrect: None
    args = test_args._replace(eta_vec=None)
    with pytest.raises(ValueError):
        oscill_angles_of_hkls(*args)

