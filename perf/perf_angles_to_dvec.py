#!/usr/bin/env python

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
    vectors = np.random.rand(n,3) - 0.5
    angles = np.random.rand(n)*twopi 
    angs = (np.random.rand(n,3) - 0.5)*twopi # (n, 3) of theta, eta, omega
    beam_vec = np.r_[0.0, 0.0, 1.0] # (3,) typically (0,0,1.)
    eta_vec = np.r_[0.0, 1.0, 0.0] # (3,) used to build an orthonormal base with beam_vec
    chi = 1.0 # scalar
    rmat_c = rot_matrix(vectors, angles)
    
    return (angs, beam_vec, eta_vec, chi, rmat_c)


def run_test():
    global sample_args
    count = 20000000
    sample_args = build_args(count)

    print(f"Running angles_to_dvec with length {count}")
    for name, module in xrd_transforms.implementations.items():
        if (name in ('numba', 'numpy')):
            continue
        
        setup_str = (f"from __main__ import sample_args, xrd_transforms\n"
                     f"fn = xrd_transforms.implementations['{name}'].angles_to_dvec")

        results = timeit.repeat(stmt=f"fn(*sample_args)",
                                setup=setup_str, repeat=3, number=3)
        print(f"{name:12}: {min(results):8.4} secs")

    del sample_args
    

if __name__ == '__main__':
    run_test()
