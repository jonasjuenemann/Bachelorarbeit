import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from numba import jit, prange, njit

num_cores = multiprocessing.cpu_count()


def findNew(y, grid):
    N = grid.shape[1]
    array = np.full(N, 0, dtype=np.int8)
    # print("Working on x[" + str(x) + "] in row " + str(y) + ".")
    for x in range(N):
        z = grid[(y - 1 + N) % N][(x + 1 + N) % N] + grid[(y - 1 + N) % N][(x + N) % N] \
            + grid[(y - 1 + N) % N][(x - 1 + N) % N] + grid[(y + N) % N][(x + 1 + N) % N] + \
            grid[(y + N) % N][(x - 1 + N) % N] + grid[(y + 1 + N) % N][(x + 1 + N) % N] + \
            grid[(y + 1 + N) % N][(x + N) % N] + grid[(y + 1 + N) % N][(x - 1 + N) % N]
        if z == 3:
            array[x] = 1
            continue
        if (grid[y][x] == 1) and (z == 2):
            array[x] = 1
            continue
    return array


def gameOfLifeJoblib(grid):
    Size = grid.shape[0]
    grid_out = np.empty_like(grid)
    grid_out = Parallel(num_cores - 1, verbose=0)(delayed(findNew)(y, grid) for y in range(Size))
    grid_out = np.array(grid_out).reshape(Size, Size)
    return grid_out


@njit(parallel=True, nogil=True)
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

if __name__ == '__main__':
    iterations = 200
    N = 1024
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
    grid = gameOfLifePara(grid)
    gameOfLifePara.parallel_diagnostics(level=4)