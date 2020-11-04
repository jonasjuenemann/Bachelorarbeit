#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np

def gameOfLife(grid, numIterations):
    cdef int N = grid.shape[0]
    grid_out = np.empty_like(grid)
    cdef int z = 0
    cdef int y = 0
    cdef int x = 0
    for i in range(numIterations):
        for y in range(N):
            for x in range(N):
                z = grid[(y - 1 + N) % N][(x + 1 + N) % N] + grid[(y - 1 + N) % N][(x + N) % N] \
                    + grid[(y - 1 + N) % N][(x - 1 + N) % N] + grid[(y + N) % N][(x + 1 + N) % N] + \
                    grid[(y + N) % N][(x - 1 + N) % N] + grid[(y + 1 + N) % N][(x + 1 + N) % N] + \
                    grid[(y + 1 + N) % N][(x + N) % N] + grid[(y + 1 + N) % N][(x - 1 + N) % N]
                grid_out[y][x] = 0
                if z == 3:
                    grid_out[y][x] = 1
                    continue
                elif (grid[y][x] == 1) and (z == 2):
                    grid_out[y][x] = 1
                    continue
        grid[:] = grid_out[:]
    return grid
