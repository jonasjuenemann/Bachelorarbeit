import sys
from time import time
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np

"""Bei dem PyCUDA Kernel wurde sich, wie schon in Kapitel 4 beschrieben, bei Tuomanen, 2018 orientiert."""

""" Der Kernel selbst besteht aus drei Schritten. Zuerst wird zu Identifikation der Threads ein X und Y Parameter gebildet. 
Anschließend wird die Breite des Arrays bestimmt, um den korrekten Index eines Threads zu identifizieren.
Die Funktion zur Identifikation der Nachbarn ist bekannt, ist nur jetzt als device FUnktion gekennzeichnet.
Der Kernel selbst besitzt dann vor allem keine Schleifen mehr, da über das Array automatisch indiziert wird."""
GoL = SourceModule("""
#define _X  (threadIdx.x + blockIdx.x * blockDim.x)
#define _Y  (threadIdx.y + blockIdx.y * blockDim.y)
#define _width  (blockDim.x * gridDim.x)
#define _true(x)  ((x + _width) % _width)
#define _index(x,y)  (_true(x) + _true(y) * _width)

__device__ int neighbors(int x, int y, int * in) {
     return ( in[_index(x -1, y+1)] + in[_index(x-1, y)] + in[_index(x-1, y-1)] \
                   + in[_index(x, y+1)] + in[_index(x, y - 1)] \
                   + in[_index(x+1, y+1)] + in[_index(x+1, y)] + in[_index(x+1, y-1)]);
}

__global__ void gameoflife(int * grid_out, int * grid) {
    int x = _X, y = _Y;
    int n = neighbors(x, y, grid);
    if (grid[_index(x,y)] == 1) {
        if (n == 2 || n == 3)  {
            grid_out[_index(x,y)] = 1;
        }
        else {
            grid_out[_index(x,y)] = 0;
        }
    }
    else if( grid[_index(x,y)] == 0 )
         if (n == 3)  {
            grid_out[_index(x,y)] = 1;
        }
        else {
            grid_out[_index(x,y)] = 0;
        }
}
""")

# Macht den Kernel für Python zugänglich
gameoflife = GoL.get_function("gameoflife")

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
    print("Durchfuehrung mit Arraygroesse: " + str(N))
    # Iteration über die Anzahl der Durchführungen
    for i in range(numDurchf):
        t_start = time()
        grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
        # Kopieren der Arrays zur GPU
        grid_gpu = gpuarray.to_gpu(grid)
        emptygrid_gpu = gpuarray.empty_like(grid_gpu)
        # Iteration des Game of Life
        for i in range(iterations):
            gameoflife(emptygrid_gpu, grid_gpu, block=(32, 32, 1), grid=(N / 32, N / 32, 1))
            # Kopieren der neuen Ergebnisse auf das originale Array
            # So kann der Kernel einfach weiterbenutzt werden.
            grid_gpu[:] = emptygrid_gpu[:]
        # Kopieren des Arrays zum Host
        grid = grid_gpu.get()
        t_end = time()
        Time += t_end - t_start
    # Bildung des Durchschnitts
    print('Total time: %fs' % (Time / numDurchf))
