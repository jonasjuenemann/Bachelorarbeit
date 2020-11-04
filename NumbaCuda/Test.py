from time import time

from numba import cuda
import numpy as np


@cuda.jit
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1


def increment_by_one_ser(an_array):
    for i in range(len(an_array)):
        an_array[i] += 1

def increment_by_one_ser_better(an_array):
    return an_array+1

iterations = 100
an_array = np.float32(np.random.random(100000))
print(an_array)
t_start = time()
for i in range(iterations):
    increment_by_one_ser(an_array)
print(an_array)
t_end = time()
print("end")
print('Total time with CPU: %f' % (t_end - t_start))
an_array = np.float32(np.random.random(100000))
print(an_array)
print(an_array)
t_start = time()
for i in range(iterations):
    an_array = increment_by_one_ser_better(an_array)
print(an_array)
t_end = time()
print("end")
print('Total time with CPU optimized: %f' % (t_end - t_start))
an_array = np.float32(np.random.random(100000))
print(an_array)
t_start = time()
threadsperblock = 32
blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock
for i in range(iterations):
    increment_by_one[blockspergrid, threadsperblock](an_array)
print(an_array)
t_end = time()
print("end")
print('Total time with cuda: %f' % (t_end - t_start))

"""
[0.5242361  0.5773896  0.3670625  ... 0.6125101  0.17399938 0.18750297]
[100.52424  100.57739  100.367065 ... 100.61251  100.173996 100.1875  ]
end
Total time with CPU: 22.945333
[0.5112034  0.29283756 0.08578198 ... 0.08111642 0.49448618 0.77292335]
[0.5112034  0.29283756 0.08578198 ... 0.08111642 0.49448618 0.77292335]
[100.5112   100.29284  100.085785 ... 100.081116 100.494484 100.77292 ]
end
Total time with CPU optimized: 0.001999
[0.47089064 0.7880578  0.71106637 ... 0.4527752  0.1594519  0.9072045 ]
[100.47089  100.788055 100.71107  ... 100.452774 100.159454 100.9072  ]
end
Total time with cuda: 0.355757
"""