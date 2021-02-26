from time import time
import numpy as np
np.random.seed(0)

def true(x, N):
    return (x + N) % N


def Nachbarn(x, y, grid, N):
    return grid[true(y - 1, N)][true(x + 1, N)] + grid[true(y - 1, N)][true(x, N)] + grid[true(y - 1, N)][true(x - 1, N)] + grid[true(y, N)][
        true(x + 1, N)] + \
           grid[true(y, N)][true(x - 1, N)] + grid[true(y + 1, N)][true(x + 1, N)] + grid[true(y + 1, N)][true(x, N)] + grid[true(y + 1,N)][
               true(x - 1, N)]


def gameOfLife(grid):
    grid_out = np.zeros_like(grid)
    cdef int Size = len(grid)
    cdef int y, x, z = 0
    for y in range(Size):
        # print("y = " + str(y))
        for x in range(Size):
            z = Nachbarn(x, y, grid, Size)
            if z == 3:
                grid_out[y][x] = 1
                continue
            elif (grid[y][x] == 1) and (z == 2):
                grid_out[y][x] = 1
            # print("x = " + str(x))
    return grid_out


cdef int iterations = 1
cdef int N = 16
grid = np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N)

print("start")
t_start = time()

for i in range(iterations):
    grid = gameOfLife(grid)


t_end = time()

print("end")
print ('Total time: %f' % (t_end - t_start))

"""
it=1
N=256
Total time: 0.263074

it=200
N=256
Total time: 49.714543

"""