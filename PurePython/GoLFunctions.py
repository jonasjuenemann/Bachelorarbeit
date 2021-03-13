import numpy as np


"""
Die Funktion nimmt eine Koordinate des Arrays sowie die Breite(Höhe) entgegen.
Liegt die Koordinate innerhalb der Breite wird diese ohne Veränderung zurückgegeben.
Sollte die Koordinate außerhalb der Breite und damit des Array Bereichs liegen, 
wird diese durch die Modulo Funktionalität auf den Anfang des Arrays gesetzt.
"""
def trueValue(x, N):
    return (x + N) % N


"""
Die Funktion nimmt eine Position im Array über die x,y Koordinaten an, sowie das Array selbst.
Sie ermittelt die Länge des Arrays.
Anschließend werden die Werte der acht Nachbarn des Arrays nach den Regeln des Game of Life abgefragt. 
Hierbei wird der "wahre Wert" des Nachbarn zuvor bestimmt, um die Randfälle abzudecken, an denen eine solche 
Abfrage sonst außerhalb Bereichs läge.
Sie gibt eine Addition der Werte, die diese Nachbarn besitzen zurück.
"""
def neighbors(x, y, grid):
    N = len(grid)
    return grid[trueValue(y - 1, N)][trueValue(x + 1, N)] + grid[trueValue(y - 1, N)][trueValue(x, N)] \
           + grid[trueValue(y - 1, N)][trueValue(x - 1, N)] + grid[trueValue(y, N)][trueValue(x + 1, N)] + \
           grid[trueValue(y, N)][trueValue(x - 1, N)] + grid[trueValue(y + 1, N)][trueValue(x + 1, N)] + \
           grid[trueValue(y + 1, N)][trueValue(x, N)] + grid[trueValue(y + 1, N)][trueValue(x - 1, N)]

"""
Die Game of Life Fu
"""
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
