import numpy as np

"""
Der boundscheck von Cython schützt davor, zu versuchen auf Werte 
außerhalb des Arrays zuzugreifen. Da dies hier nicht gemacht wird, 
kann der Schutz deaktiviert werden. Spart ein wenig Zeit
"""
#cython: boundscheck=False

"""
Diese Funktion führt ein Game of Life auf einem als 
Parameter gegebenen (2D-)Array aus.
Das Game of Life wird dabei so oft ausgeführt, 
wie als Parameter numIterations übergeben wird.
Die Funktionsweise des GoL wurde bereits für die Python Dateien beschrieben
und wurde hier auch nicht verändert.
Neu sind die statischen Typdefinitionen zu Anfang.
"""
def gameOfLife(grid, numIterations):
    cdef int N = grid.shape[0]
    """
    Dieses Grid braucht nur einmal erstellt werden über alle Iterationen, 
    da am der Ende jeder Iterationdas originale Grid mit den aktualisierten 
    Werten überschrieben wird.
    Auf grid_help immer nur geschrieben, die Werte sind dabei egal.
    """
    grid_help = np.empty_like(grid)
    cdef int z = 0
    cdef int y = 0
    cdef int x = 0
    for i in range(numIterations):
        for y in range(N):
            for x in range(N):
                z = grid[(y - 1 + N) % N][(x + 1 + N) % N] + \
                    grid[(y - 1 + N) % N][(x + N) % N] \
                    + grid[(y - 1 + N) % N][(x - 1 + N) % N] + \
                    grid[(y + N) % N][(x + 1 + N) % N] + \
                    grid[(y + N) % N][(x - 1 + N) % N] + \
                    grid[(y + 1 + N) % N][(x + 1 + N) % N] + \
                    grid[(y + 1 + N) % N][(x + N) % N] + \
                    grid[(y + 1 + N) % N][(x - 1 + N) % N]
                grid_help[y][x] = 0
                if z == 3:
                    grid_help[y][x] = 1
                    continue
                elif (grid[y][x] == 1) and (z == 2):
                    grid_help[y][x] = 1
                    continue
        grid = np.array(grid_help, copy=True)
    return grid
