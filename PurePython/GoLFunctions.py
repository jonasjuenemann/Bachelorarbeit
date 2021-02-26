import numpy as np



def trueValue(x, N):
    return (x + N) % N

def neighbors(x, y, grid):
    N = len(grid)
    return grid[trueValue(y - 1, N)][trueValue(x + 1, N)] + grid[trueValue(y - 1, N)][trueValue(x, N)] \
           + grid[trueValue(y - 1, N)][trueValue(x - 1, N)] + grid[trueValue(y, N)][trueValue(x + 1, N)] + \
           grid[trueValue(y, N)][trueValue(x - 1, N)] + grid[trueValue(y + 1, N)][trueValue(x + 1, N)] + \
           grid[trueValue(y + 1, N)][trueValue(x, N)] + grid[trueValue(y + 1, N)][trueValue(x - 1, N)]

def gameOfLifeNaiv(grid):
    grid_out = np.empty_like(grid)
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



def gameOfLife(grid):
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
