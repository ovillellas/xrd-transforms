#!/usr/bin/env python

import sys
import timeit
import time
import numpy as np

import xrd_transforms

def rot_matrix(vector, angle):
    assert(len(vector) == len(angle))
    rmat = np.empty((len(vector), 3, 3))
    vector = xrd_transforms.unit_vector(vector)
    
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c;

    rmat[:,0,0] = c + vector[:,0]*vector[:,0]*t
    rmat[:,1,1] = c + vector[:,1]*vector[:,1]*t
    rmat[:,2,2] = c + vector[:,2]*vector[:,2]*t

    tmp1 = vector[:,0]*vector[:,1]*t
    tmp2 = vector[:,2]*s
    rmat[:,1,0] = tmp1 + tmp2
    rmat[:,0,1] = tmp1 - tmp2

    tmp1 = vector[:,0]*vector[:,2]*t
    tmp2 = vector[:,1]*s
    rmat[:,2,0] = tmp1 - tmp2
    rmat[:,0,2] = tmp1 + tmp2

    tmp1 = vector[:,1]*vector[:,2]*t
    tmp2 = vector[:,0]*s
    rmat[:,2,1] = tmp1 + tmp2
    rmat[:,1,2]  = tmp1 - tmp2

    return rmat
    

def build_args(n):
    twopi = 2*np.pi
    angs = (np.random.rand(n,3) - 0.5)*twopi # (n, 3) of theta, eta, omega
    vector = np.random.rand(1,3) - 0.5
    angle = np.random.rand(1)*twopi 
    beam_vec = np.r_[0.0, 0.0, 1.0] # (3,) typically (0,0,1.)
    eta_vec = np.r_[0.0, 1.0, 0.0] # (3,) used to build an orthonormal base with beam_vec
    chi = 1.0 # scalar
    rmat_c = rot_matrix(vector, angle)
    
    return (angs, beam_vec, eta_vec, chi, rmat_c[0])


def run_test(counts=[20000000]):
    global run_args

    max_count = max(counts)
    sample_args = build_args(max_count)

    test_count = max(counts)
    print(f'Checking results with length {test_count}')
    angs, beam_vec, eta_vec, chi, rmat_c = sample_args
    test_args = (angs[0:test_count,:], beam_vec, eta_vec, chi, rmat_c)
    ref_result = xrd_transforms.implementations['capi'].angles_to_dvec(*test_args)
    for name, module in xrd_transforms.implementations.items():
        if (name in ('numpy','numba')):
            continue
        result = module.angles_to_dvec(*test_args)
        np.testing.assert_allclose(result, ref_result, rtol=1e-5, atol=1e-6);
        isOk = 'PASS' if np.allclose(result, ref_result, rtol=1e-5, atol=1e-6) else 'FAIL'
        print(f'Implementation {name}: {isOk}')

    for count in counts:
        print(f'Running angles_to_dvec with length {count}')
        run_args = (angs[0:count,:], beam_vec, eta_vec, chi, rmat_c)
        for name, module in xrd_transforms.implementations.items():
            if (name in ('numba', 'numpy')):
                continue

            setup_str = (f"from __main__ import run_args, xrd_transforms\n"
                         f"fn = xrd_transforms.implementations['{name}'].angles_to_dvec")

            results = timeit.repeat(stmt=f"fn(*run_args)",
                                    setup=setup_str, repeat=3, number=3)
            print(f"{name:12}: {min(results):8.4} secs")

    del sample_args
    

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) > 1:
        args = [int(arg) for arg in sys.argv[1:]]
        run_test(counts=args)
    else:
        run_test()
