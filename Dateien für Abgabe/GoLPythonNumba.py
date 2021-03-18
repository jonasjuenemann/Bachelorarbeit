import numpy as np
from numba import jit, prange

"""Game of Life Funktion in Python mit Numba. Wie zuvor wird ein Parameter grid als 2D Array entgegengenommen, das Game of Life auf diesem durchgeführt 
und anschließend ein überarbeitetes Array zurückgegeben. Neu ist der Numba Dekorator. Der @jit Dekorator zeigt numba an, das die Funktion Just-in-Time kompiliert werden soll. 
Es wird außerdem der Parameter nopython im Dekorator übergeben, sodass eine Ausführung ohne Einmischung des Python Interpreters vorgenommen werden kann."""


@jit(nopython=True)
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


"""Für eine parallele Verarbeitung des Game of Life kann außerdem der Parameter Parallel=true im Dekorator ergänzt werden. 
Der Dekorator @njit stellt dabei eine Abkürzung des zuvor verwendeten @jit(nopython=True) Dekorators dar, bedeutet in der Ausführung aber dasselbe. 
Wenn True, versucht nogil, den GIL innerhalb der kompilierten Funktion freizugeben. Der GIL wird nur freigegeben, wenn Numba die Funktion im nopython-Modus kompilieren kann, 
andernfalls wird eine Kompilierungswarnung ausgegeben."""


@njit(parallel=True, nogil=True)
def gameOfLifeParallel(grid):
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
