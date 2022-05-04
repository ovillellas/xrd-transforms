# tests for the reference implementation of ray-plane intersection
# note this function may not be used directly, but it is a reference that can
# be used by other tests to check results. In a interim way it could be used
# in reference implementations as well.
#
# As such, this tests won't need to use parametrization to check different
# implementations, as the entry points would need.

from __future__ import absolute_import

import pytest
import numpy as np
from numpy.testing import assert_allclose

from xrd_transforms import reference as xf_ref

def plane_from_point_and_normal(point, normal):
    # note: we use a plane representations of (A, B, C, D) where
    # Ax + By + Cz - D = 0 is the plane equation.
    D = point @ normal
    return np.hstack((normal, D))

def assert_point_in_plane(pt, plane):
    assert_allclose(pt@plane[:3]-plane[3], 0.0, atol=1e-07)

def test_trivial():
    '''test a simple case, including the expected result of t for the intersection'''
    pp = np.r_[0.0, 4.0, 0.0] # plane at (0,4,0)
    pn = np.r_[0.0, -1.0, 0.0] # with normal (0,-1,0)
    rp = np.r_[0.0, 0.0, 0.0] # ray at origin
    rv = np.r_[0.0, 1.0, 0.0] # pointing towards (0,1,0)
    plane = plane_from_point_and_normal(pp, pn)

    expected = 4.0
    t = xf_ref.intersect_ray_plane(rp, rv, plane)
    assert t==expected
    assert_point_in_plane(rp+t*rv, plane)

    rv2 = np.r_[0.0, 2.0, 0.0] # non identity...
    expected2 = 2.0 # t is scaled according to the vector magnitude
    t2 = xf_ref.intersect_ray_plane(rp, rv2, plane)
    assert t2==expected2
    assert_point_in_plane(rp+t2*rv2, plane)


def test_non_parallel():
    '''test a sample non-trivial plane, with a sample non-trivial ray'''
    pp = np.r_[3.0, 2.0, 0.0]
    pn = np.r_[0.5, 1.0, -0.5] # no need the plane normal to be normalized
    rp = np.r_[-1.0, 0.0, -2.0]
    rv = np.r_[0.3, 0.3, 0.3] # no need the ray vector to be normalized
    plane = plane_from_point_and_normal(pp, pn)

    t = xf_ref.intersect_ray_plane(rp, rv, plane)
    assert_point_in_plane(rp+t*rv,plane)


def test_singular():
    '''test singular cases where there is no intersection'''
    pp = np.r_[0.0, 4.0, 0.0]
    pn = np.r_[0.0, -1.0, 0.0] # no need the plane normal to be normalized
    rp = np.r_[0.0, 0.0, 0.0]
    rv = np.r_[0.0, 0.0, 1.0] # no need the ray vector to be normalized
    plane = plane_from_point_and_normal(pp, pn)

    # no intersections as rv and pn are perpendicular
    t = xf_ref.intersect_ray_plane(rp, rv, plane)
    assert not np.isfinite(t)
    # in fact, this should be an infinity as the point is not in the plane
    assert np.isinf(t)

    # similar, but with the ray being in the plane
    t = xf_ref.intersect_ray_plane(pp, rv, plane)
    assert not np.isfinite(t)
    # in fact... this should result in a nan
    assert np.isnan(t)
