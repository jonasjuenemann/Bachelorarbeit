import numpy as np

"""
Diese Funktion führt ein Game of Life auf einem als 
Parameter gegebenen (2D-)Array aus.
Im Gegensatz zur vorherigen Implementation wird hier
nicht modularisiert
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
def gameOfLife(grid):
    grid_out = np.empty_like(grid)
    N = grid.shape[0]
    for y in range(N):
        for x in range(N):
            z = grid[(y - 1 + N) % N][(x + 1 + N) % N] + grid[(y - 1 + N) % N][(x + N) % N] + \
                grid[(y - 1 + N) % N][(x - 1 + N) % N] + grid[(y + N) % N][(x + 1 + N) % N] + \
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