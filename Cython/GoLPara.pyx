#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
from cython.parallel cimport prange

"""
schwierig, prange funktioniert nur ohne gil, aber ohne gil kein Access auf Python Objekte erlaubt. ich braeuchte also c arrays hier.



Parallelize with Cython over numpy:


import numpy as np

def foo():
    cdef int i, j, n

    x = np.zeros((200, 2000), float)

    n = x.shape[0]
    for i in prange(n, nogil=True):
        with gil:
            for j in range(100):
                x[i,:] = np.cos(x[i,:])

    return x


prange() takes a few other arguments including num_threads: which will default to the number of cores on your system and schedule: which has to do with load balancing. 
The simplest option here is ‘static’ which just breaks the loop into equal chunks. This is fine if all the steps compute in around the same time.
"""

cdef gameOfLife(grid):
    grid_out = np.empty_like(grid)
    cdef int N = grid.shape[0]
    cdef int y = 0
    cdef int x = 0
    cdef int z = 0
    for y in range(N):
        for x in range(N):
            z = grid[(y - 1 + N) % N][(x + 1 + N) % N] + grid[(y - 1 + N) % N][(x + N) % N] \
                + grid[(y - 1 + N) % N][(x - 1 + N) % N] + grid[(y + N) % N][(x + 1 + N) % N] + \
                grid[(y + N) % N][(x - 1 + N) % N] + grid[(y + 1 + N) % N][(x + 1 + N) % N] + \
                grid[(y + 1 + N) % N][(x + N) % N] + grid[(y + 1 + N) % N][(x - 1 + N) % N]
            grid_out[y][x] = 0
            if z == 3:
                grid_out[y][x] = 1
            if (grid[y][x] == 1) and (z == 2):
                grid_out[y][x] = 1
    return grid_out
