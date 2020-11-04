import math
from time import time
import numba
from numba import cuda
import numpy as np

np.random.seed(0)

"""
Even though Numba can automatically transfer NumPy arrays to the device, it can only do so conservatively by always transferring device memory
back to the host when a kernel finishes. To avoid the unnecessary transfer for read-only arrays, you can use the following APIs to manually control the transfer:
"""


@cuda.jit
def gameOfLife_ker(array_out, array_in):
    x, y = cuda.grid(ndim=2)
    N = cuda.blockDim.x * cuda.gridDim.x
    if x < N and y < N:
        z = array_in[(y - 1 + N) % N][(x + 1 + N) % N] + array_in[(y - 1 + N) % N][(x + N) % N] \
            + array_in[(y - 1 + N) % N][(x - 1 + N) % N] + array_in[(y + N) % N][(x + 1 + N) % N] + \
            array_in[(y + N) % N][(x - 1 + N) % N] + array_in[(y + 1 + N) % N][(x + 1 + N) % N] + \
            array_in[(y + 1 + N) % N][(x + N) % N] + array_in[(y + 1 + N) % N][(x - 1 + N) % N]
        array_out[y][x] = 0
        if z == 3:
            array_out[y][x] = 1
        if (array_in[y][x] == 1) and (z == 2):
            array_out[y][x] = 1


if __name__ == '__main__':
    N = 16384
    iterations = 200
    t_start = time()
    an_array = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
    print(an_array)
    an_array_gpu = numba.cuda.to_device(an_array)
    new_array_gpu = numba.cuda.device_array_like(an_array_gpu)
    block = (32, 32)
    grid = (int(N / 32), int(N / 32))
    for i in range(iterations):
        gameOfLife_ker[grid, block](new_array_gpu, an_array_gpu)
        an_array_gpu[:] = new_array_gpu[:]
    print(an_array_gpu.copy_to_host())
    t_end = time()
    print("end")
    print('Total time: %fs' % (t_end - t_start))
