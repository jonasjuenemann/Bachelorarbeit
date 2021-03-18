from timeit import Timer
import numpy as np
from GoLPython import gameOfLife

"""Die Funktion nimmt eine Game of Life Funktion, einen Parameter für 
die Iterationen und die geteste Größe des Arrays entgegen.
Dabei wird ein Ausgangszustand hergeleitet mit einem entsprechendem NumPy
Kommando.
Anschließend wird das Game of Life für die Anzahl der Iterationen durchgeführt.
"""


def gameOfLifeTimer(func, i, N):
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
    for i in range(i):
        grid = func(grid)


"""
Für jede beliebige Game of Life Funktion kann die Dauer ermittelt werden.
Iteriert wird dabei über zu testenden Arraygrößen.
Hierfür wird der Timer von dem timeit von Python verwendet.
die Anzahl der Durchführungen kann vorher festgelegt werden.
Über diese wird ein Durchschnitt gebildet.
Anschließend die geteste Arragröße sowie die
durchschnittliche Dauer per Konsole ausgegeben.
"""

if __name__ == '__main__':

    """Die getesteten Arraygrößen"""
    X = [32, 64, 128, 256, 512, 1024, 2048]
    # Y = [1024, 2048, 4096, 8192, 16384]
    """Anzahl der getesteten Iterationen"""
    iterations = 20
    """Anzahl der Durchführungen aus der der Durchschnitt
    gebildet wird"""
    anzahlDurchfuehrungen = 100

    for i in X:
        t = Timer(lambda: gameOfLifeTimer(gameOfLife, iterations, i))
        avgtime = t.timeit(number=anzahlDurchfuehrungen) / anzahlDurchfuehrungen
        print("Durchführung mit Arraygröße: " + str(i))
        print('Total time gameOfLife: %f' % (avgtime))
