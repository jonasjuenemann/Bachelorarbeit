from time import time
from multiprocessing.pool import ThreadPool

import numpy as np

pool = ThreadPool()

def findNew(x, y):
    global grid
    # print("Working on x[" + str(x) + "] in row " + str(y) + ".")
    z = grid[(y - 1 + N) % N][(x + 1 + N) % N] + grid[(y - 1 + N) % N][(x + N) % N] \
        + grid[(y - 1 + N) % N][(x - 1 + N) % N] + grid[(y + N) % N][(x + 1 + N) % N] + \
        grid[(y + N) % N][(x - 1 + N) % N] + grid[(y + 1 + N) % N][(x + 1 + N) % N] + \
        grid[(y + 1 + N) % N][(x + N) % N] + grid[(y + 1 + N) % N][(x - 1 + N) % N]
    if z == 3:
        return 1
    if (grid[y][x] == 1) and (z == 2):
        return 1
    return 0


def gameOfLifeRow(y):
    global grid
    global grid_out
    Size = len(grid)
    grid_out[y] = [findNew(x, y) for x in range(Size)]


if __name__ == '__main__':
    np.random.seed(0)
    iterations = 20
    N = 256
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
    grid_out = np.empty_like(grid)
    print("started with gridsize " + str(N) + " and " + str(iterations) + " iterations")
    t_start = time()
    print(grid)
    for i in range(iterations):
        pool.map(gameOfLifeRow, range(N))
        grid = grid_out
    print(grid)
    t_end = time()
    print("end")
    print('Total time: %f' % (t_end - t_start))


    """
[[0 0 0 0 0 0 0 1 0 0]
 [0 0 0 0 1 0 0 0 0 0]
 [0 0 0 0 0 0 1 0 0 0]
 [0 0 1 0 0 0 1 1 0 0]
 [0 0 0 1 0 0 0 0 1 0]
 [0 0 1 1 1 0 1 0 1 0]
 [0 1 0 0 0 0 1 0 0 1]
 [1 1 0 0 1 0 0 0 0 0]
 [0 0 1 1 0 0 1 1 0 0]
 [0 0 0 0 0 0 0 1 0 0]]
 
    
    """
