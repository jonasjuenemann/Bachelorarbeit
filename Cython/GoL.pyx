from cython.parallel import prange
import numpy as np
cimport numpy as np

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
Dabei wird zunächst eine "leere" Kopie des Arrays erstellt 
und die Breite des Arrays ermittelt.
Anschließend wird mit einer doppelten for-Schleife über das 
Array iteriert. Dabei wird für jede Zelle betrachtet, 
wie viele lebendige Nachbarn diese Zelle hat.
Bei drei Nachbarn wird der Punkt mit einer 1 markiert 
(bleibt lebendig), bei zwei bleibt die Zelle Lebendig,
wenn diese schon zuvor lebendig ist. In jedem anderen 
Fall stirbt die Zelle (wird mit einer 0 markiert)
Die neuen Werte werden dabei in dem zu Anfang erstellten, 
neuen Array gespeichert. Das übergegebene Array wird 
zu keinem Zeitpunkt in der Funktion geändert.
Am Ende gibt die Funktion das Array mit den neuen Werten zurück.
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
                z = grid[(y - 1 + N) % N][(x + 1 + N) % N] + grid[(y - 1 + N) % N][(x + N) % N] \
                    + grid[(y - 1 + N) % N][(x - 1 + N) % N] + grid[(y + N) % N][(x + 1 + N) % N] + \
                    grid[(y + N) % N][(x - 1 + N) % N] + grid[(y + 1 + N) % N][(x + 1 + N) % N] + \
                    grid[(y + 1 + N) % N][(x + N) % N] + grid[(y + 1 + N) % N][(x - 1 + N) % N]
                grid_help[y][x] = 0
                if z == 3:
                    grid_help[y][x] = 1
                    continue
                elif (grid[y][x] == 1) and (z == 2):
                    grid_help[y][x] = 1
                    continue
        grid = np.array(grid_help, copy=True)
    return grid
