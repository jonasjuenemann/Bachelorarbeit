import numpy as np
from numba import jit, prange

"""
fuer Parallelisierung koennte man hier @jit(Parallel=True) hinzuefuegen, das gaebe nicht nur keinen Speedup, es waere signifikant langsamer.
0.42 -> 1.1 sec.

# https://stackoverflow.com/questions/50658884/why-this-numba-code-is-6x-slower-than-numpy-code
"""


@jit(nopython=True, nogil=True, fastmath=True)
def gameOfLifeJit(grid):
    grid_out = np.empty_like(grid)
    N = grid.shape[0]
    for y in range(N):
        for x in range(N):
            z = grid[(y - 1 + N) % N][(x + 1 + N) % N] + grid[(y - 1 + N) % N][(x + N) % N] \
                + grid[(y - 1 + N) % N][(x - 1 + N) % N] + grid[(y + N) % N][(x + 1 + N) % N] + \
                grid[(y + N) % N][(x - 1 + N) % N] + grid[(y + 1 + N) % N][(x + 1 + N) % N] + \
                grid[(y + 1 + N) % N][(x + N) % N] + grid[(y + 1 + N) % N][(x - 1 + N) % N]
            if z == 3:
                grid_out[y][x] = 1
                continue
            if (grid[y][x] == 1) and (z == 2):
                grid_out[y][x] = 1
                continue
            grid_out[y][x] = 0
    return grid_out


@jit(parallel=True)
def gameOfLifePara(grid):
    grid_out = np.empty_like(grid)
    N = grid.shape[0]
    for y in prange(N):
        for x in range(N):
            z = grid[(y - 1 + N) % N][(x + 1 + N) % N] + grid[(y - 1 + N) % N][(x + N) % N] \
                + grid[(y - 1 + N) % N][(x - 1 + N) % N] + grid[(y + N) % N][(x + 1 + N) % N] + \
                grid[(y + N) % N][(x - 1 + N) % N] + grid[(y + 1 + N) % N][(x + 1 + N) % N] + \
                grid[(y + 1 + N) % N][(x + N) % N] + grid[(y + 1 + N) % N][(x - 1 + N) % N]
            if z == 3:
                grid_out[y][x] = 1
                continue
            if (grid[y][x] == 1) and (z == 2):
                grid_out[y][x] = 1
                continue
            grid_out[y][x] = 0
    return grid_out

"""
if __name__ == '__main__':
    np.random.seed(0)
    iterations = 10
    N = 256
    t_start = time()
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))

    print("started with gridsize " + str(N) + " and " + str(iterations) + " iterations")

    for i in range(iterations):
        grid = gameOfLifeJit(grid)

    t_end = time()
    print("end")
    print('Total time with jit: %f' % (t_end - t_start))

    t_start = time()
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))

    "Hier sollte nicht, grid als global Variable in der Funktion benutzt werden, viel langsamer!"
    print("started with gridsize " + str(N) + " and " + str(iterations) + " iterations")

    for i in range(iterations):
        grid = gameOfLifePara(grid)

    t_end = time()
    print("end")
    print('Total time with jit in parallel: %f' % (t_end - t_start))
"""