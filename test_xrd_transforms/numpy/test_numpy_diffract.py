# unit-test for diffraction code in the numpy implementation.
#
# diffraction has been moved out to a simple function that takes a batch
# of gvecs and optionally a beam vector (it defaults to the standard one).
#
# having it as a function allows some simple unit-testing of it.

from __future__ import absolute_import

from collections import namedtuple

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises
from xrd_transforms import xf_numpy


Experiment = namedtuple('Experiment', ['gvecs'])



@pytest.fixture(scope='module')
def experiment():
    '''Note this fixture is data is actually extracted from some test runs.

    It just provides several gvecs, the first 8 are along the FRAME axes. The
    other 8 are just in the middle of each different octant of the FRAME.
    '''
    sample = np.sqrt(1.0/3.0)
    yield Experiment(
        gvecs=np.array([[ 0.0,  0.0,  1.0],
                        [ 0.0,  0.0, -1.0],
                        [ 0.0,  1.0,  0.0],
                        [ 0.0, -1.0,  0.0],
                        [ 1.0,  0.0,  0.0],
                        [-1.0,  0.0,  0.0],
                        [ sample, sample, sample],
                        [ sample, sample,-sample],
                        [ sample,-sample, sample],
                        [ sample,-sample,-sample],
                        [-sample, sample, sample],
                        [-sample, sample,-sample],
                        [-sample,-sample, sample],
                        [-sample,-sample,-sample]])
        )


def test_vectorization_no_beam(experiment):
    '''check that vectorized works as many calls to single'''
    vector_result = xf_numpy.diffract(experiment.gvecs)
    scalar_results = np.empty_like(experiment.gvecs)
    for i, gvec in enumerate(experiment.gvecs):
        scalar_results[i] = xf_numpy.diffract(gvec)

    assert_equal(vector_result, scalar_results)


def test_vectorization_beam(experiment):
    '''check that vectorized works as many calls to single'''
    beam = np.r_[0.0, 1.0, 0.0]
    vector_result = xf_numpy.diffract(experiment.gvecs, beam)
    scalar_results = np.empty_like(experiment.gvecs)
    for i, gvec in enumerate(experiment.gvecs):
        scalar_results[i] = xf_numpy.diffract(gvec, beam)

    assert_equal(vector_result, scalar_results)


def test_default_value(experiment):
    '''check that an explicit [0, 0, -1] beam is equivalent to the implicit beam'''
    beam = np.r_[0.0, 0.0, -1.0]
    gvecs = experiment.gvecs
    a = xf_numpy.diffract(gvecs)
    b = xf_numpy.diffract(gvecs, beam)
    print(f'a = {a}\nb = {b}\n')
    assert_equal(a, b)


def test_no_diffract(experiment):
    '''Some cases where diffraction should no happen'''
    nans = np.r_[np.nan, np.nan, np.nan]

    # note default beam is (0, 0, -1)

    # completely aligned gvec and beam.
    assert_equal(xf_numpy.diffract(np.r_[0.0, 0.0, -1.0]), nans)
    assert_equal(xf_numpy.diffract(np.r_[0.0, 0.0, 1.0]), nans)

    # orthogonal gvec and beam
    assert_equal(xf_numpy.diffract(np.r_[0.0, 1.0, 0.0]), nans)

    # in range
    sample = np.sqrt(1.0/3.0)
    assert_equal(xf_numpy.diffract(np.r_[sample, sample, -sample]), nans)

    with assert_raises(AssertionError):
        assert_equal(xf_numpy.diffract(np.r_[sample, sample, sample]), nans)
