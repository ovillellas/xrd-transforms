# tests for rays_to_xy_planar

from __future__ import absolute_import

from collections import namedtuple

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_raises

from common import function_implementations


all_impls = pytest.mark.parametrize('rays_to_xy_planar_impl, module_name',
                                    function_implementations('rays_to_xy_planar'))

Experiment = namedtuple('Experiment', ['vectors', 'origins', 'rmat_d', 'tvec_d',
                                       'expected'])

@pytest.fixture(scope='module')
def experiment():
    '''a very simple setup, but should yield unique rays for all combinations'''
    yield Experiment(
        vectors = np.array([[0.0, 0.0, 1.0],
                            [0.0, 0.0, -1.0]]),
        origins = np.array([[[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
                            [[0.1, 0.0, 0.0], [0.1, 0.1, 0.0]],
                            [[-0.1, 0.0, 0.0], [-0.1, 0.1, 0.0]],
                            [[0.2, 0.0, 0.0], [0.2, 0.1, 0.0]]]),
        rmat_d = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]]),
        tvec_d = np.r_[0.0,0.0,3.0],
        expected = np.array([[[0.0, 0.0], [np.nan, np.nan]],
                             [[0.1, 0.0], [np.nan, np.nan]],
                             [[-0.1, 0.0], [np.nan, np.nan]],
                             [[0.2, 0.0], [np.nan, np.nan]]])
        )

##############################################################################
# Test arguments and result dimensions.
##############################################################################
def dimensional_checker(fn):
    '''helper function that will pass to the function fn as many np.zeros arguments
    as shapes are provided as inputs. It will return the shape of the results.

    This can help the test the argument dimension checks by just providing the
    wanted dimensions, with filler data. Some tests are there just to check the
    supported shapes.

    This is slightly different in the way it handles the results compared to
    similar code in test_gvec_to_rays.

    '''
    def _dim_checker(dims, **kw):
        results = fn(*map(np.zeros, dims), **kw)
        return results.shape
    return _dim_checker


@all_impls
def test_result_dimensions(rays_to_xy_planar_impl, module_name):
    checker = dimensional_checker(rays_to_xy_planar_impl)

    # (1,1) batch of rays. Origin per vector should be indifferent in this case
    r = checker(((3,), (3,), (3,3), (3,)), origin_per_vector=False)
    assert r==(2,)

    r = checker(((3,), (3,), (3,3), (3,)), origin_per_vector=True)
    assert r==(2,)

    # (M,1) batch of rays (different origins, same vector).
    # Origin per vector still indifferent
    r = checker(((3,), (4,3), (3,3), (3,)), origin_per_vector=False)
    assert r==(4,2)
    r = checker(((3,), (4,3), (3,3), (3,)), origin_per_vector=True)
    assert r==(4,2)

    # (1,N) batch of rays (same origin, different vectors).
    # Origin per vector becomes relevant.
    r = checker(((2,3), (3,), (3,3), (3,)), origin_per_vector=False)
    assert r==(2,2)
    r = checker(((2,3), (2,3), (3,3), (3,)), origin_per_vector=True)
    assert r==(2,2)

    # (M,N) batch of rays (different origins, different vectors)
    # Origin per vector also relevant in this case
    r = checker(((2,3), (4,3), (3,3), (3,)), origin_per_vector=False)
    assert r==(4,2,2)
    r = checker(((2,3), (4,2,3), (3,3), (3,)), origin_per_vector=True)
    assert r==(4,2,2)


@all_impls
def test_args_vectors(rays_to_xy_planar_impl, module_name):
    # note: vectors is the first arg
    checker = dimensional_checker(rays_to_xy_planar_impl)

    # vectors inner dimension must be 3
    with assert_raises(ValueError):
        checker(((2,), (3,), (3,3), (3,)), origin_per_vector=False)

    with assert_raises(ValueError):
        checker(((2,2), (3,), (3,3), (3,)), origin_per_vector=False)

    # too few dimensions
    with assert_raises(ValueError):
        checker((tuple(), (3,), (3,3), (3,)), origin_per_vector=False)

    # too many dimensions
    with assert_raises(ValueError):
        checker(((2,2,3), (3,), (3,3), (3,)), origin_per_vector=False)

@all_impls
def test_args_origins(rays_to_xy_planar_impl, module_name):
    # note: origins is the second arg
    checker = dimensional_checker(rays_to_xy_planar_impl)

    # origins inner dimension must be 3
    with assert_raises(ValueError):
        checker(((3,), (2,), (3,3), (3,)), origin_per_vector=False)

    with assert_raises(ValueError):
        checker(((3,), (4,2), (3,3), (3,)), origin_per_vector=False) # M=4, N=-

    with assert_raises(ValueError):
        checker(((2,3), (2,2), (3,3), (3,)), origin_per_vector=True) # M=-, N=2

    with assert_raises(ValueError):
        checker(((2,3), (4,2,2,), (3,3), (3,)), origin_per_vector=True) # M=4, N=2

    # too few dimensions
    with assert_raises(ValueError):
        checker(((3,), tuple(), (3,3), (3,)), origin_per_vector=False)

    with assert_raises(ValueError):
        checker(((2,3), (3,), (3,3), (3,)), origin_per_vector=True)

    # too many dimensions
    with assert_raises(ValueError):
        checker(((3,), (5,4,3), (3,3), (3,)), origin_per_vector=False)

    with assert_raises(ValueError):
        checker(((2,3), (5,4,2,3), (3,3), (3,)), origin_per_vector=True)

    # origins-vectors dimension mismatch
    with assert_raises(ValueError):
        checker(((2, 3), (4, 3, 3), (3,3), (3,)), origin_per_vector=True)

    with assert_raises(ValueError):
        checker(((2,3), (2, 4, 3), (3,3), (3,)), origin_per_vector=True)


@all_impls
def test_args_rmat_d(rays_to_xy_planar_impl, module_name):
    # note: rmat_d is the third arg
    checker = dimensional_checker(rays_to_xy_planar_impl)

    # rmat_d must be a (3,3) matrix
    with assert_raises(ValueError):
        checker(((3,), (3,), (2,2), (3,)), origin_per_vector=False)

    # too few dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,), (3,), (3,)), origin_per_vector=False)

    # too many dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,), (5,3,3), (3,)), origin_per_vector=False)


@all_impls
def test_args_tvec_d(rays_to_xy_planar_impl, module_name):
    # note: tvec_d is the fourth arg
    checker = dimensional_checker(rays_to_xy_planar_impl)

    # tvec_d must be a 3-vector
    with assert_raises(ValueError):
        checker(((3,), (3,), (3,3), (2,)), origin_per_vector=False)

    # too few dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,), (3,3), tuple()), origin_per_vector=False)

    # too many dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,), (3,3), (3,3)), origin_per_vector=False)

##############################################################################
# Vectorization testing
# remember this is potentially a NxM problem, with N being the number of
# vectors for the rays, and M the number of voxels.
##############################################################################
@all_impls
def test_N_vectorization(experiment, rays_to_xy_planar_impl, module_name):
    # test for origin_per_vector=False
    vectors = experiment.vectors
    origins = experiment.origins[0,0,:] # just use one of the vectors
    rmat_d = experiment.rmat_d
    tvec_d = experiment.tvec_d

    vectorized_results = rays_to_xy_planar_impl(vectors, origins,
                                                rmat_d, tvec_d,
                                                origin_per_vector=False)

    iterated_results = np.empty_like(vectorized_results)
    for i in range(len(vectors)):
        iterated_results[i,:] = rays_to_xy_planar_impl(vectors[i], origins,
                                                       rmat_d, tvec_d,
                                                       origin_per_vector=False)
    assert_allclose(vectorized_results, iterated_results)


@all_impls
def test_N_vectorization_opv(experiment, rays_to_xy_planar_impl, module_name):
    # test for origin_per_vector=True
    vectors = experiment.vectors
    origins = experiment.origins[0,...]
    rmat_d = experiment.rmat_d
    tvec_d = experiment.tvec_d

    vectorized_results = rays_to_xy_planar_impl(vectors, origins,
                                                rmat_d, tvec_d,
                                                origin_per_vector=True)

    iterated_results = np.empty_like(vectorized_results)
    for i in range(len(vectors)):
        iterated_results[i,:] = rays_to_xy_planar_impl(vectors[i], origins[i],
                                                       rmat_d, tvec_d,
                                                       origin_per_vector=False)
    assert_allclose(vectorized_results, iterated_results)


@all_impls
def test_M_vectorization(experiment, rays_to_xy_planar_impl, module_name):
    vectors = experiment.vectors[0]
    origins = experiment.origins[:,0,:] # just use one of the vectors
    rmat_d = experiment.rmat_d
    tvec_d = experiment.tvec_d

    vectorized_results = rays_to_xy_planar_impl(vectors, origins,
                                                rmat_d, tvec_d,
                                                origin_per_vector=False)

    iterated_results = np.empty_like(vectorized_results)
    for i in range(len(origins)):
        iterated_results[i,:] = rays_to_xy_planar_impl(vectors, origins[i],
                                                       rmat_d, tvec_d,
                                                       origin_per_vector=False)
    assert_allclose(vectorized_results, iterated_results)


@all_impls
def test_MN_vectorization(experiment, rays_to_xy_planar_impl, module_name):
    # for origins_per_vector=False
    vectors = experiment.vectors
    origins = experiment.origins[:,0,:] # just use one of the vectors
    rmat_d = experiment.rmat_d
    tvec_d = experiment.tvec_d

    vectorized_results = rays_to_xy_planar_impl(vectors, origins,
                                                rmat_d, tvec_d,
                                                origin_per_vector=False)

    iterated_results = np.empty_like(vectorized_results)
    for n_i in range(len(vectors)):
        for m_i in range(len(origins)):
            iterated_results[m_i, n_i,:] = rays_to_xy_planar_impl(vectors[n_i], origins[m_i],
                                                                  rmat_d, tvec_d,
                                                                  origin_per_vector=False)
    assert_allclose(vectorized_results, iterated_results)


@all_impls
def test_MN_vectorization_opv(experiment, rays_to_xy_planar_impl, module_name):
    # for origins_per_vector=True
    vectors = experiment.vectors
    origins = experiment.origins
    rmat_d = experiment.rmat_d
    tvec_d = experiment.tvec_d

    vectorized_results = rays_to_xy_planar_impl(vectors, origins,
                                                rmat_d, tvec_d,
                                                origin_per_vector=True)

    iterated_results = np.empty_like(vectorized_results)
    for n_i in range(len(vectors)):
        for m_i in range(len(origins)):
            iterated_results[m_i, n_i,:] = rays_to_xy_planar_impl(vectors[n_i], origins[m_i, n_i],
                                                                  rmat_d, tvec_d,
                                                                  origin_per_vector=False)
    assert_allclose(vectorized_results, iterated_results)

##############################################################################
# Minimal functional testing
# For some trivial values. More thorough tests would be needed.
# As vectorization is supposed to work at this point if vectorization tests
# pass, the tests can make use of vectorization or just tests single cases
# with discrete calls.
##############################################################################

@all_impls
def test_sample_cases(experiment, rays_to_xy_planar_impl, module_name):
    vectors = experiment.vectors
    origins = experiment.origins
    rmat_d = experiment.rmat_d
    tvec_d = experiment.tvec_d

    results = rays_to_xy_planar_impl(vectors, origins,
                                     rmat_d, tvec_d,
                                     origin_per_vector=True)
    assert_allclose(results, experiment.expected)
