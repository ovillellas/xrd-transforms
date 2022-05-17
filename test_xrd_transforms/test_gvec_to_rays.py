# tests for gvec_to_rays

from __future__ import absolute_import

from collections import namedtuple

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_raises
from common import function_implementations

# these are used in some test
from xrd_transforms import unit_vector
from xrd_transforms.xf_numpy import diffract


all_impls = pytest.mark.parametrize('gvec_to_rays_impl, module_name',
                                    function_implementations('gvec_to_rays'))


# gvec_to_rays uses a reduced argument list wrt the previous gvec_to_xy. This is
# due to being 'detector independent'.
Experiment = namedtuple('Experiment', ['rtol', 'gvec_c', 'rmat_s', 'rmat_c',
                                       'tvec_s', 'tvec_c'])


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
def dimensional_checker(fn):
    '''helper function that will pass to the function fn as many np.zeros arguments
    as shapes are provided as inputs. It will return the shape of the results.

    This can help the test the argument dimension checks by just providing the
    wanted dimensions, with filler data. Some tests are there just to check the
    supported shapes.
    '''
    def _dim_checker(dims):
        results = fn(*map(np.zeros, dims))
        return (r.shape for r in results)
    return _dim_checker


@all_impls
def test_result_dimensions(gvec_to_rays_impl, module_name):
    # gvec_c is the first argument, rmat_s
    checker = dimensional_checker(gvec_to_rays_impl)

    # one gvec, one voxel
    v, p = checker(((3,), (3,3), (3,3), (3,), (3,)))
    assert v==(3,) and p==(3,)

    # one gvec, many voxels
    v, p = checker(((3,), (3,3), (3,3), (3,), (4,3)))
    assert v==(3,) and p==(4,3)

    # many gvecs, single rmat_s, single voxel
    v, p = checker(((2,3), (3,3), (3,3), (3,), (3,)))
    assert v==(2,3) and p==(3,)

    # many gvecs, rmat_s per gvec, single voxel
    v, p = checker(((2,3), (2,3,3), (3,3), (3,), (3,)))
    assert v==(2,3) and p == (2,3) # origin point per gvec (as rmat_s varies)

    # many gvecs, single rmat_s, many voxels
    v, p = checker(((2,3), (3,3), (3,3), (3,), (4,3)))
    assert v==(2,3) and p==(4,3) # origin point per voxel

    # many gvecs, rmat_s per gvec, many voxels
    v, p = checker(((2,3), (2,3,3), (3,3), (3,), (4,3)))
    assert v==(2,3) and p==(4,2,3) # origin point per voxel and per gvec, as rmat_s varies


@all_impls
def test_args_gvecs(gvec_to_rays_impl, module_name):
    # note: gvecs is the first arg
    checker = dimensional_checker(gvec_to_rays_impl)
    # gvecs inner dimension should be 3
    with assert_raises(ValueError):
        checker(((4,2),(3,3),(3,3),(3,),(3,)))

    # gvecs can't have more dimensions than a single dimension of 3-vectors
    with assert_raises(ValueError):
        checker(((3,3,3),(3,3),(3,3),(3,),(3,)))


@all_impls
def test_args_rmat_s(gvec_to_rays_impl, module_name):
    # note: rmat_s is the second arg
    checker = dimensional_checker(gvec_to_rays_impl)
    # single rmat_s, must be (3,3)
    with assert_raises(ValueError):
        checker(((3,), (3,4), (3,3), (3,), (3,)))

    # missing dims in rmat_s
    with assert_raises(ValueError):
        checker(((3,), (3,), (3,3), (3,), (3,)))

    # N mismatch between gvecs and rmat_s
    with assert_raises(ValueError):
        checker(((2,3), (4,3,3), (3,3), (3,), (3,)))

    # No support for many rmat_s over a single gvec
    with assert_raises(ValueError):
        checker(((3,), (4,3,3), (3,3), (3,), (3,)))

    # Multiple dimensions for vectorized gvecs/rmat_s not supported
    with assert_raises(ValueError):
        checker(((2,4,3), (2,4,3,3), (3,3), (3,), (3,)))


@all_impls
def test_args_rmat_c(gvec_to_rays_impl, module_name):
    # rmat_c is only supported to be a (3,3) COB matrix
    # note: rmat_c is the third arg
    checker = dimensional_checker(gvec_to_rays_impl)

    # bad dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (4,4), (3,), (3,)))

    # too few dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,), (3,), (3,)))

    # too many dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,3,3), (3,), (3,)))


@all_impls
def test_args_tvec_s(gvec_to_rays_impl, module_name):
    # tvec_s is only supported to be a single (3,) translation vector
    # note: tvec_s is the fourth arg
    checker = dimensional_checker(gvec_to_rays_impl)

    # bad dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,3), (4,), (3,)))

    # too few dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,3), tuple(), (3,)))

    # too many dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,3), (2,3), (3,)))

@all_impls
def test_args_tvec_c(gvec_to_rays_impl, module_name):
    # tvec_c can be either (3,) or (M, 3), depending on the number of voxels
    # note: tvec_c is the fifth arg
    checker = dimensional_checker(gvec_to_rays_impl)

    # bad dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,3), (3,), (4,)))

    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,3), (3,), (2, 4,)))

    # too few dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,3), (3,), tuple()))

    # too many dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,3), (3,), (3, 4, 3)))


@all_impls
def test_args_beam(gvec_to_rays_impl, module_name):
    # beam is an optional kw argument that defaults to the standard beam vector
    # in our setup. If present it should be a single 3-vector (shape (3,))
    # note: beam is the optional seventh arg
    checker = dimensional_checker(gvec_to_rays_impl)

    # bad dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,3), (3,), (3,), (2,)))

    # too few dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,3), (3,), (3,), tuple()))

    # too many dimensions
    with assert_raises(ValueError):
        checker(((3,), (3,3), (3,3), (3,), (3,), (2,3)))


##############################################################################
# Vectorization tests. These test checks vectorization works as intended
##############################################################################

@all_impls
def test_N_vectorization_single_rmat_s(experiment, gvec_to_rays_impl, module_name):
    # check that vectorized call for gvec is equivalent to N calls to
    # non-vectorized
    gvec_c = experiment.gvec_c
    rmat_s = experiment.rmat_s[0] # a single rmat_s
    rmat_c = experiment.rmat_c
    tvec_s = experiment.tvec_s
    tvec_c = experiment.tvec_c[0] # a single tvec_c

    # vectored result
    v, p = gvec_to_rays_impl(gvec_c, rmat_s, rmat_c, tvec_s, tvec_c)

    # scalar results.
    v2 = np.empty_like(v)
    for i, gvec in enumerate(gvec_c):
        v2[i,:], p2 = gvec_to_rays_impl(gvec, rmat_s, rmat_c, tvec_s, tvec_c)
        # origin positions should be the same for all gvec
        assert_allclose(p, p2)
    assert_allclose(v, v2)


@all_impls
def test_N_vectorization_many_rmat_s(experiment, gvec_to_rays_impl, module_name):
    # check that vectorized call for gvec is equivalent to N calls to
    # non-vectorized
    gvec_c = experiment.gvec_c
    rmat_s = experiment.rmat_s
    rmat_c = experiment.rmat_c
    tvec_s = experiment.tvec_s
    tvec_c = experiment.tvec_c[0] # a single tvec_c

    # vectored result
    v, p = gvec_to_rays_impl(gvec_c, rmat_s, rmat_c, tvec_s, tvec_c)

    # scalar results. Generates as many points as gvecs
    v2 = np.empty_like(v)
    p2 = np.empty_like(p)
    for i, gvec in enumerate(gvec_c):
        v2[i,:], p2[i,:] = gvec_to_rays_impl(gvec_c[i], rmat_s[i], rmat_c, tvec_s, tvec_c)

    assert_allclose(p, p2)
    assert_allclose(v, v2)


@all_impls
def test_M_vectorization(experiment, gvec_to_rays_impl, module_name):
    # check that vectorized call for tvec_c is equivalent to M calls to
    # non-vectorized
    gvec_c = experiment.gvec_c[0]
    rmat_s = experiment.rmat_s[0]
    rmat_c = experiment.rmat_c
    tvec_s = experiment.tvec_s
    tvec_c = experiment.tvec_c

    # vectored result
    v, p = gvec_to_rays_impl(gvec_c, rmat_s, rmat_c, tvec_s, tvec_c)

    # scalar results. Generates as many points as gvecs
    p2 = np.empty_like(p)
    for i, voxel in enumerate(tvec_c):
        v2, p2[i,:] = gvec_to_rays_impl(gvec_c, rmat_s, rmat_c, tvec_s, tvec_c[i])
        assert_allclose(v, v2)
    assert_allclose(p, p2)


@all_impls
def test_NxM_vectorization_single_rmat_s(experiment, gvec_to_rays_impl, module_name):
    # check that vectorized call for gvec and tvec is equivalent to MxN calls to
    # non-vectorized
    gvec_c = experiment.gvec_c
    rmat_s = experiment.rmat_s[0]
    rmat_c = experiment.rmat_c
    tvec_s = experiment.tvec_s
    tvec_c = experiment.tvec_c

    # vectored result
    v, p = gvec_to_rays_impl(gvec_c, rmat_s, rmat_c, tvec_s, tvec_c)

    # scalar results. Generates as many points as gvecs
    v2 = np.empty_like(v)
    p2 = np.empty_like(p)
    for m_i, _ in enumerate(tvec_c):
        for n_i, _ in enumerate(gvec_c):
            v2[n_i,:], p2[m_i,:] = gvec_to_rays_impl(gvec_c[n_i], rmat_s, rmat_c,
                                                     tvec_s, tvec_c[m_i])
            # checks are made in the inner_loop to check that results are stable
            # even for those that in theory should not change.
            assert_allclose(v2[n_i], v[n_i])
            assert_allclose(p2[m_i], p[m_i])


@all_impls
def test_NxM_vectorization_many_rmat_s(experiment, gvec_to_rays_impl, module_name):
    # check that vectorized call for gvec and tvec is equivalent to MxN calls to
    # non-vectorized
    gvec_c = experiment.gvec_c
    rmat_s = experiment.rmat_s
    rmat_c = experiment.rmat_c
    tvec_s = experiment.tvec_s
    tvec_c = experiment.tvec_c

    # vectored result
    v, p = gvec_to_rays_impl(gvec_c, rmat_s, rmat_c, tvec_s, tvec_c)

    # scalar results. Generates as many points as gvecs
    v2 = np.empty_like(v)
    p2 = np.empty_like(p)
    for m_i, _ in enumerate(tvec_c):
        for n_i, _ in enumerate(gvec_c):
            v2[n_i,:], p2[m_i, n_i, :] = gvec_to_rays_impl(gvec_c[n_i], rmat_s[n_i], rmat_c,
                                                           tvec_s, tvec_c[m_i])
            # checks are made in the inner_loop to check that results are stable
            # even for those that in theory should not change.
            assert_allclose(v2[n_i], v[n_i])
            assert_allclose(p2[m_i, n_i], p[m_i, n_i])


##############################################################################
# Functional tests would be desirable with a set of known results.
# These can be done a single case at a time, as the vectorization is already
# tested.
##############################################################################

@all_impls
def test_functional_example(gvec_to_rays_impl, module_name):
    gvec_c = np.r_[0.0, 0.0, 1.0]
    rmat_s = np.eye(3)
    rmat_c = np.eye(3)
    tvec_s = np.r_[0.0, 0.0, 1.0]
    tvec_c = np.r_[0.0, 0.0, 0.01]

    v, p = gvec_to_rays_impl(gvec_c, rmat_s, rmat_c, tvec_s, tvec_c)

    assert_allclose(p, np.r_[0.0, 0.0, 1.01]) # with all identities, p = tvec_s+tvec_c
    assert_allclose(v, np.r_[np.nan, np.nan, np.nan]) # gvec is orthogonal to beam


@all_impls
def test_functional_example2(gvec_to_rays_impl, module_name):
    gvec_c = unit_vector(np.r_[0.0, 1.0, 1.0])
    rmat_s = np.eye(3)
    rmat_c = np.eye(3)
    tvec_s = np.r_[0.0, 0.0, 1.0]
    tvec_c = np.r_[0.0, 0.0, 0.01]

    v, p = gvec_to_rays_impl(gvec_c, rmat_s, rmat_c, tvec_s, tvec_c)

    assert_allclose(p, np.r_[0.0, 0.0, 1.01]) # with all identities, p = tvec_s+tvec_c
    assert_allclose(v, diffract(gvec_c)) # with all identity matrices, v should be straight diffraction of the gvec
