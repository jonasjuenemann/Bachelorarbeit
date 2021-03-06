import math
from time import time
import numba
from numba import cuda
import numpy as np

np.random.seed(0)

"""Kernel zur Ausführung des GoL. Besitzt wie bei PyCUDA eine automatische Indizierung"""
@cuda.jit
def gameOfLife_ker(array_out, array_in):
    # Thread Indizes können komfortabel automatisch identifiziert werden.
    x, y = cuda.grid(ndim=2)
    # Die Arraygröße entspricht der Größe eines Blocks * der Menge der Blöcke
    N = cuda.blockDim.x * cuda.gridDim.x
    # Für alle Punkte im Array wird dann das GoL berechnet
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

    X = [32, 64, 128, 256, 512, 1024, 2048]
    Y = [1024, 2048, 4096, 8192, 16384]
    iterations = 20
    numDurchf = 10
    for z in Y:
        Time = 0
        N = z
        print("Durchführung mit Arraygröße: " + str(N))
        for i in range(numDurchf):
            t_start = time()
            an_array = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
            an_array_gpu = numba.cuda.to_device(an_array)
            array_gpu_out = numba.cuda.device_array_like(an_array_gpu)
            block = (32, 32)
            grid = (int(N / 32), int(N / 32))
            for i in range(iterations):
                gameOfLife_ker[grid, block](array_gpu_out, an_array_gpu)
                an_array_gpu[:] = array_gpu_out[:]
            an_array = an_array_gpu.copy_to_host()
            t_end = time()
            Time += t_end - t_start
        print("end")
        print('Total time: %fs' % (Time / numDurchf))
