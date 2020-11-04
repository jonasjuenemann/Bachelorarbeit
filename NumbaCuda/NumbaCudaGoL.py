from time import time
import numba
from numba import cuda
import numpy as np
#import sys

#np.set_printoptions(threshold=sys.maxsize)
np.random.seed(0)

"""
Even though Numba can automatically transfer NumPy arrays to the device, it can only do so conservatively by always transferring device memory
back to the host when a kernel finishes. To avoid the unnecessary transfer for read-only arrays, you can use the following APIs to manually control the transfer:
"""


@cuda.jit(device=True)
def trueValue(x, N):
    return (x + N) % N




"""
@cuda.jit(device=True)
def Neighbors(x, y, array, N):
    return array[Index(x - 1, y + 1, N)] + array[Index(x - 1, y, N)] + array[Index(x - 1, y - 1, N)] \
           + array[Index(x, y + 1, N)] + array[Index(x, y - 1, N)] \
           + array[Index(x + 1, y + 1, N)] + array[Index(x + 1, y, N)] + array[Index(x + 1, y - 1, N)]
"""

@cuda.jit
def gameOfLife_ker(array_out, array_in):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    N = cuda.blockDim.x * cuda.gridDim.x
    if x < N and y < N:
        z = array_in[(x - 1 + N) % N, (y + 1 + N) % N] + array_in[(x - 1 + N) % N, y] + array_in[(x - 1 + N) % N, (y - 1 + N) % N] \
            + array_in[x, (y + 1 + N) % N] + array_in[x, (y - 1 + N) % N] \
            + array_in[(x + 1 + N) % N, (y + 1 + N) % N] + array_in[(x + 1 + N) % N, y] + array_in[(x + 1 + N) % N, (y - 1 + N) % N]
        array_out[x, y] = 0
        if z == 3:
            array_out[x, y] = 1
        if (array_in[x, y] == 1) and (z == 2):
            array_out[x, y] = 1


if __name__ == '__main__':
    N = 4096
    iterations = 200
    t_start = time()
    an_array = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
    print(an_array)
    an_array_gpu = numba.cuda.to_device(an_array)
    new_array_gpu = numba.cuda.device_array_like(an_array_gpu)
    block = (32, 32)
    grid = (int(N // 32), int(N // 32))
    for i in range(iterations):
        gameOfLife_ker[grid, block](new_array_gpu, an_array_gpu)
        an_array_gpu[:] = new_array_gpu[:]
    an_array = an_array_gpu.copy_to_host()
    print(an_array)
    t_end = time()
    print("end")
    print('Total time: %fs' % (t_end - t_start))

    """
[[0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 1 1 1 0 0 1 0 0]
 [1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 1 1]
 [1 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 1]
 [0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0]
 [1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1 1]
 [1 0 0 0 0 0 0 0 0 0 1 1 0 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1]
 [1 0 0 0 0 1 1 1 1 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0]
 [1 0 0 1 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1]
 [0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 1 1 0 1 0 0 0]
 [0 0 1 1 0 1 0 1 0 0 0 1 0 0 0 0 0 1 1 1 1 0 0 1 0 1 0 0 1 0 0 0]
 [0 0 1 0 1 0 1 1 1 1 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 1 1 0 0 0 0 0]
 [0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0]
 [0 1 1 0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 0 1 0 1 1 0 0]
 [0 1 1 1 1 1 0 1 1 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0]
 [1 0 1 0 1 1 1 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 1]
 [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0]
 [0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0]
 [1 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 1 1 0 0 0]
 [1 0 0 0 0 1 0 1 1 0 1 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 1 0 0]
 [0 0 0 0 1 1 1 0 1 1 0 0 0 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 1 0 0 0]
 [0 0 0 0 0 1 0 0 0 0 0 1 1 1 0 0 1 0 1 0 0 1 1 0 1 0 0 0 1 0 0 0]
 [0 1 1 0 1 0 1 0 0 1 1 0 1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [1 1 0 0 1 0 1 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [0 1 0 0 0 0 1 1 0 0 1 1 0 0 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0]
 [0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 1 0 1 0 0 0 0 1 0]
 [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 1 0 0 1 0 0 0 1 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 1 0 0 0 1 1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0]
 [1 1 1 0 0 0 1 1 1 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
 [1 0 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1 0 0 0 0 0 0 1 0 1 0 1 0 1 0 1]
 [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1]]
    """