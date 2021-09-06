#!/usr/bin/env python

'''Module to test the performance of different implementations of gvec_to_xy.

gvec_to_xy performs the forward simulation from the gvecs to the detector space
(paramatric coordinates). This function is critical in nf-HEDM, for example
'''
import timeit
import time
import argparse
import numpy as np

import xrd_transforms

def some_fancy_generator_of_unit_vectors(count):
    """return a (COUNT,3) array of COUNT normalized 3-vectors"""
    # not that fancy, but data values should not matter
    result = np.zeros((count,3))
    result[:,1] = 1.0
    return result

def some_fancy_generator_of_rotation_matrices(count):
    # not that fancy...
    result = np.empty((count, 3, 3))
    result[:] = np.eye(3)
    return result

def build_args(args):
    '''build some sample args for gvec_to_xy. Anything will do as long as it is
    appropriate.
    gvec_to_xy requires the following:

    gvecs (COUNT, 3) array. NCOUNT gvecs
    rmat_d (3,3) array. LAB to DETECTOR rotation matrix.
    rmat_s ([COUNT,]3,3) array. LAB to SAMPLE rotation matrices (1 or COUNT)
    rmat_c (3,3) array. SAMPLE to CRYSTAL rotation matrix
    tvec_d (3,) array. LAB to DETECTOR translation vector.
    tvec_s (3,) array. LAB to SAMPLE translation vector.
    tvec_c (3,) array. SAMPLE to CRYSTAL translation vector.

    depending on the 'single_s' argument, either 1 (if present) or COUNT
    rmat_s will be generated.
    '''
    count = args.count
    gvecs = some_fancy_generator_of_unit_vectors(count)
    rmat_d = np.eye(3) # as in test suite
    if not args.single_s:
        rmat_s = some_fancy_generator_of_rotation_matrices(count)
    else:
        # taken from test suite
        rmat_s = np.array([[ 0.77029942,  0.        ,  0.63768237],
                           [ 0.        ,  1.        , -0.        ],
                           [-0.63768237,  0.        ,  0.77029942]])

    rmat_c = np.array([[ 0.91734473, -0.08166131,  0.38962815],
                       [ 0.31547749,  0.74606417, -0.58639766],

                       [-0.24280159,  0.66084771,  0.71016033]]) # as in test suite
    # the following as in test_suite
    tvec_d = np.array([ 0. ,  1.5, -5. ])
    tvec_s = np.array([0., 0., 0.])
    tvec_c = np.array([-0.25, -0.25, -0.25])

    # return all as a tuple for convenience when calling...
    return (gvecs, rmat_d, rmat_s, rmat_c, tvec_d, tvec_s, tvec_c)


def run_test(args):
    global sample_args
    impl_to_run = build_impls(args.impl)

    sample_args = build_args(args)

    print(f"Running gvec_to_xy with COUNT={args.count} single rMat_s={args.single_s}")
    for name in impl_to_run:
        setup_str = (f"from __main__ import sample_args, xrd_transforms\n"
                     f"fn = xrd_transforms.implementations['{name}'].gvec_to_xy")

        results = timeit.repeat(stmt=f"fn(*sample_args)",
                                setup=setup_str, repeat=3, number=3)
        print(f"{name:12}: {min(results):8.4} secs")

    del sample_args

def build_impls(requested):
    '''builds a list of implementations that will be run. Preserve order.
    If 'all' is encountered, all implementations containing the gvec_to_xy will
    be added to the list in some order.'''
    available = available_impls()
    result = []
    for impl in requested:
        if impl == 'all':
            result.extend([im for im in available if im not in result])
            continue
        if impl not in result:
            if impl in available:
                result.append(impl)
            else:
                print(f"Implementation '{impl}' is not available.")

    return result

def available_impls():
    return set([name for name, module in xrd_transforms.implementations.items()
                if hasattr(module, 'gvec_to_xy')])

def show_impls():
    with_gvec_to_xy = [name for name, module in xrd_transforms.implementations.items()
                       if hasattr(module, 'gvec_to_xy')]
    without_gvec_to_xy =  [name for name, module in xrd_transforms.implementations.items()
                       if not hasattr(module, 'gvec_to_xy')]
    
    if len(with_gvec_to_xy):
        str_impls = "', '".join(with_gvec_to_xy)
        print(f'gvec_to_xy implementations: \'{str_impls}\'.')
    else:
        print(f'No available gvec_to_xy implementation (!)')

    if len(without_gvec_to_xy):
        str_impls = "'. '".join(without_gvec_to_xy)
        print(f'modules with no gvec_to_xy implementations: \'{str_impls}\'.')
    else:
        print(f'all available implementations have a gvec_to_xy implementation')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--impl', action='append', default=[],
                        help='Add implementation to run. Use "all" for all')
    parser.add_argument('--count', type=int, default=500,
                        help='Number of gvecs to use')
    parser.add_argument('--single-s', action='store_true',
                        help='Use a single SAMPLE rotation matrix for all gvecs')
    parser.add_argument('--available', action='store_true',
                        help='Show available implementations')
    args = parser.parse_args()
    # ['count', 'impl', 'single_s', 'available']
    return args
    
if __name__ == '__main__':
    args = parse_args()
    if args.available:
        show_impls()
    else:
        run_test(args)

