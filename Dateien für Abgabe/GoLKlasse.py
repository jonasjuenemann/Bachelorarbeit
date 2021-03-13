import numpy as np
"""
Die Implementation des GoL als Klasse
Die Funktionalitäten sind dieselben wie in der nicht Objektorientierten 
Variante mit Funktionen.
Die Implementation ist zwar etwas eleganter, war aber in den Performance-
Tests deutlich langsamer.
"""
class GameOfLife:
    def __init__(self, N):
        self.N = N
        self.grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))

    def trueValue(self, x):
        return (x + self.N) % self.N

    def neighbors(self, x, y):
        return self.grid[self.trueValue(y - 1)][self.trueValue(x + 1)] + \
               self.grid[self.trueValue(y - 1)][self.trueValue(x)] + \
               self.grid[self.trueValue(y - 1)][self.trueValue(x - 1)] + \
               self.grid[self.trueValue(y)][self.trueValue(x + 1)] + \
               self.grid[self.trueValue(y)][self.trueValue(x - 1)] + \
               self.grid[self.trueValue(y + 1)][self.trueValue(x + 1)] + \
               self.grid[self.trueValue(y + 1)][self.trueValue(x)] + \
               self.grid[self.trueValue(y + 1)][self.trueValue(x - 1)]

    def gameOfLife(self):
        grid_out = np.empty_like(self.grid)
        for y in range(self.N):
            for x in range(self.N):
                z = self.neighbors(x, y)
                if z == 3:
                    grid_out[y][x] = 1
                    continue
                if (self.grid[y][x] == 1) and (z == 2):
                    grid_out[y][x] = 1
                    continue
                grid_out[y][x] = 0
        self.grid = grid_out

    def printGoL(self):
        print(self.grid)

