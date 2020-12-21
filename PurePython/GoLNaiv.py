import copy
import random
from time import time

import numpy as np

random.seed(0)


def trueValue(x, N):
    return (x + N) % N


def neighbors(x, y, grid):
    N = len(grid)
    return grid[trueValue(y - 1, N)][trueValue(x + 1, N)] + grid[trueValue(y - 1, N)][trueValue(x, N)] \
           + grid[trueValue(y - 1, N)][trueValue(x - 1, N)] + grid[trueValue(y, N)][trueValue(x + 1, N)] + \
           grid[trueValue(y, N)][trueValue(x - 1, N)] + grid[trueValue(y + 1, N)][trueValue(x + 1, N)] + \
           grid[trueValue(y + 1, N)][trueValue(x, N)] + grid[trueValue(y + 1, N)][trueValue(x - 1, N)]


def gameOfLife(grid):
    N = len(grid)
    grid_out = []
    for y in range(N):
        list = []
        for x in range(N):
            z = neighbors(x, y, grid)
            print(f"Für {y} und {x} ist der z-wert bei {z}")
            if z == 3:
                list.append(1)
                continue
            if (grid[y][x] == 1) and (z == 2):
                list.append(1)
                continue
            list.append(0)
        grid_out.append(list)
    return grid_out

#funktioniert nicht, da er an den flaschen Stellen auf das grid zugreift und dieses verändert (er soll aber nur grid_out verändern) -> deswegen deepcopy
def gameOfLifeLists(grid):
    grid_out = copy.deepcopy(grid)
    N = len(grid)
    for y in range(N):
        for x in range(N):
            z = neighbors(x, y, grid)
            if z == 3:
                grid_out[y][x] = 1
                continue
            if (grid[y][x] == 1) and (z == 2):
                grid_out[y][x] = 1
                continue
            grid_out[y][x] = 0
    return grid_out


def printgrid(grid):
    for y in grid:
        print(y)


if __name__ == '__main__':
    X = [32, 64, 128, 256, 512, 1024, 2048]
    iterations = 20
    N = X[5]
    p = 0.75  # Verteilung von 0 und 1 im Array

    print("started with gridsize " + str(N) + " and " + str(iterations) + " iterations")

    t_start = time()

    grid = []

    for y in range(N):
        list = []
        for x in range(N):
            k = random.random()
            if k > p:
                list.append(1)
                continue
            list.append(0)
        grid.append(list)

    # printgrid(grid)

    for i in range(iterations):
        grid = gameOfLifeLists(grid)
        # print()
        # printgrid(grid)

    t_end = time()
    print("end")
    print('Total time naiv: %f' % (t_end - t_start))

    """
    generiert wesentlich schneller
    
    t_start = time()
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))

    t_end = time()
    print("end")
    print('Total time optimierter: %f' % (t_end - t_start))

    """
