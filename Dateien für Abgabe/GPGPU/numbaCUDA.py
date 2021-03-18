from time import time
import numba
from numba import cuda
import numpy as np

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


# Der Kernel muss nicht explizit zugänglich gemacht werden.

if __name__ == '__main__':
    # Zu testende Arraygrößen
    X = [32, 64, 128, 256, 512, 1024, 2048]
    Y = [1024, 2048, 4096, 8192, 16384]
    # Anzahl Iterationen bzw. Durchführungen
    iterations = 20
    numDurchf = 100

    # Iteration über die Arraygrößen
    for z in Y:
        Time = 0
        N = z
        print("Durchführung mit Arraygröße: " + str(N))
        # Iteration über die Anzahl der Durchführungen
        for i in range(numDurchf):
            t_start = time()
            an_array = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
            # Kopieren der Arrays zur GPU
            an_array_gpu = numba.cuda.to_device(an_array)
            array_gpu_out = numba.cuda.device_array_like(an_array_gpu)
            # Iteration des Game of Life
            for i in range(iterations):
                gameOfLife_ker[32, 32, int(N / 32), int(N / 32)](array_gpu_out, an_array_gpu)
                # Kopieren der neuen Ergebnisse auf das originale Array
                # So kann der Kernel einfach weiterbenutzt werden.
                an_array_gpu[:] = array_gpu_out[:]
            # Kopieren des Arrays zum Host
            an_array = an_array_gpu.copy_to_host()
            t_end = time()
            Time += t_end - t_start
        # Bildung des Durchschnitts
        print('Total time: %fs' % (Time / numDurchf))
