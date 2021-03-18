import numpy as np

"""
Die Funktion nimmt eine Koordinate des Arrays sowie die Breite(Höhe) als Parameter entgegen.
Liegt die Koordinate innerhalb der Breite wird diese ohne Veränderung zurückgegeben.
Sollte die Koordinate außerhalb der Breite und damit des Array Bereichs liegen, 
wird diese durch die Modulo Funktionalität auf den Anfang des Arrays gesetzt.
Die Funktion gibt die ermittelte Koordinate zurück.
"""


def trueValue(x, N):
    return (x + N) % N


"""
Die Funktion nimmt eine Position im Array über die x,y Koordinaten, sowie das Array selbst als Parameter an.
Sie ermittelt zunächst die Länge des Arrays.
Anschließend werden die Werte der acht Nachbarn des Arrays nach den Regeln des Game of Life abgefragt. 
Hierbei wird der "wahre Wert" des Nachbarn zuvor bestimmt, um die Randfälle abzudecken, an denen eine solche 
Abfrage sonst außerhalb Bereichs läge.
Sie gibt eine Addition der Werte, die diese Nachbarn besitzen, zurück.
"""


def neighbors(x, y, grid):
    N = len(grid)
    return grid[trueValue(y - 1, N)][trueValue(x + 1, N)] + \
           grid[trueValue(y - 1, N)][trueValue(x, N)] \
           + grid[trueValue(y - 1, N)][trueValue(x - 1, N)] + \
           grid[trueValue(y, N)][trueValue(x + 1, N)] + \
           grid[trueValue(y, N)][trueValue(x - 1, N)] + \
           grid[trueValue(y + 1, N)][trueValue(x + 1, N)] + \
           grid[trueValue(y + 1, N)][trueValue(x, N)] + \
           grid[trueValue(y + 1, N)][trueValue(x - 1, N)]


"""
Diese Funktion führt ein Game of Life auf einem als Parameter gegebenen (2D-)Array aus.
Dabei wird zunächst eine "leere" Kopie des Arrays erstellt und die Breite des Arrays ermittelt.
Anschließend wird mit einer doppelten for-Schleife über das Array iteriert.
Dabei wird für jede Zelle betrachtet, wie viele lebendige Nachbarn diese Zelle hat.
Bei drei Nachbarn wird der Punkt mit einer 1 markiert (bleibt lebendig), bei zwei bleibt die Zelle Lebendig,
wenn diese schon zuvor lebendig ist. In jedem anderen Fall stirbt die Zelle (wird mit einer 0 markiert)
Die neuen Werte werden dabei in dem zu Anfang erstellten, neuen Array gespeichert. Das übergegebene Array wird 
zu keinem Zeitpunkt in der Funktion geändert.
Am Ende gibt die Funktion das Array mit den neuen Werten zurück.
"""


def gameOfLife(grid):
    grid_out = np.empty_like(grid)
    N = len(grid)
    for y in range(N):
        for x in range(N):
            # Ermittlung der Nachbarn der Zelle
            z = neighbors(x, y, grid)
            # Bei 3 lebendigen Nachbarn lebt die Zelle in jedem Fall
            if z == 3:
                grid_out[y][x] = 1
                # Die Ermittlung für diese Zelle ist abgeschlossen.
                # Es soll mit der nächsten Zelle weitergemacht werden -> continue
                continue
            # Bei 2 lebendigen Nachbarn lebt die Zelle in dem Fall, dass Sie lebendig ist.
            if (grid[y][x] == 1) and (z == 2):
                grid_out[y][x] = 1
                continue
            # Wenn keine der vorherigen Bedingungen zutrifft, ist die Zelle tot.
            grid_out[y][x] = 0
    return grid_out
