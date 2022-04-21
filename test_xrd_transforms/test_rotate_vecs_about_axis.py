# tests for rotate_vecs_about_axis

from __future__ import absolute_import

import pytest

import numpy as np
from numpy.testing import assert_allclose

from common import function_implementations

all_impls = pytest.mark.parametrize('rotate_vecs_about_axis_impl, module_name', 
                                    function_implementations('rotate_vecs_about_axis'))


@all_impls
def test_single_angle_axis(rotate_vecs_about_axis_impl, module_name):
    angle = np.pi
    axis = np.r_[0.0, 1.0, 0.0]
    vecs = np.eye(3)
    result = rotate_vecs_about_axis_impl(angle, axis.T, vecs.T)

    # 180 degree rotation around y axis: y vector should remain
    # equal and the others should change sign
    assert_allclose(-vecs[0, :], result[0, :], atol=1e-10)
    assert_allclose(vecs[1, :], result[1, :], atol=1e-10)
    assert_allclose(-vecs[2, :], result[2, :], atol=1e-10)


@all_impls
def test_multiple_angles_single_axis(rotate_vecs_about_axis_impl, module_name):
    angles = np.r_[np.pi, 2*np.pi]
    axis = np.r_[0.0, 1.0, 0.0]
    vecs = np.array([[1.0, 1.0, 0.0],
                     [1.0, 1.0, 0.0]])
    result = rotate_vecs_about_axis_impl(angles, axis, vecs.T)
    assert_allclose(result[:, 0], np.r_[-1.0, 1.0, 0.0], atol=1e-10)
    assert_allclose(result[:, 1], np.r_[1.0, 1.0, 0.0], atol=1e-10)

    
@all_impls
def test_single_angle_multiple_axes(rotate_vecs_about_axis_impl, module_name):
    angle = np.pi
    axes = np.array([[0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])
    vecs = np.ones((2,3))

    result = rotate_vecs_about_axis_impl(angle, axes.T, vecs.T)

    assert_allclose(result[:,0], np.r_[-1.0, 1.0, -1.0], atol=1e-10)
    assert_allclose(result[:,1], np.r_[-1.0, -1.0, 1.0], atol=1e-10)


@all_impls
def test_multiple_angle_multiple_axes(rotate_vecs_about_axis_impl, module_name):
    angles = np.r_[ 0.5*np.pi, -0.5*np.pi ]
    axes = np.array([[0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])
    vecs = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0]])

    result = rotate_vecs_about_axis_impl(angles, axes.T, vecs.T)

    assert_allclose(result[:, 0], np.r_[0.0, 0.0, -1.0], atol=1e-10)
    assert_allclose(result[:, 1], np.r_[1.0, 0.0, 0.0], atol=1e-10)
