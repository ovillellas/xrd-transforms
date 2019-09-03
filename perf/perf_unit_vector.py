#!/usr/bin/env python

import timeit
import time
import numpy as np

import xrd_transforms

def run_test():
    global sample_array

    sizes = [ int(10**i) for i in range(0, 7) ]
    max_size = max(sizes)
    sample_array = np.random.rand(max_size, 3)

    for size in sizes:
        number = int(max_size / size)
        print(f"Using size: {size} times: {number} (best of 3)")
        for name, module in xrd_transforms.implementations.items():
            setup_str = (f"from __main__ import sample_array, xrd_transforms\n"
                         f"fn = xrd_transforms.implementations['{name}'].unit_vector")

            results = timeit.repeat(stmt=f"fn(sample_array[:{size},:])",
                                    setup=setup_str, timer=time.clock,
                                    repeat=3, number=number)
            print(f"{name:12}: {min(results):8.4} secs")
    
    del sample_array


if __name__ == '__main__':
    run_test()
