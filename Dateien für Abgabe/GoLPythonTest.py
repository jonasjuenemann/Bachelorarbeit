import numpy as np
from GoLPython import gameOfLife

# Lösungen für N=10 (und np.seed(0))
solutions = [[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
              [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
              [0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]],
             [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
              [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
              [0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
              [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
             ]

"""Dieser Funktion kann eine GoL Funktion als Parameter übergeben werden.
    Anschließend überprüft die Ergebnisse der übergebenen Funktion
    indem er sie zwei per hand erstellten Lösungen gegenüberstellt.
    Die Funktion gibt einen Boolean zurück, der True aussagt, wenn
    die Funktion korrekte Ergebnisse erstellt
 """


def testIfGoLCorrect(func, solutions):
    correct = True
    N = 10
    iterations = [1, 2]

    for x in range(len(iterations)):
        """
        Der Seed wird manuell festgelegt, sonst würden pseudo zufällige
        Ergebnisse entstehen. Für eine Überprüfung wäre das ungeeignet.
        """
        np.random.seed(0)
        grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
        for i in range(iterations[x]):
            grid = func(grid)
        correct = (correct and np.alltrue(grid == solutions[x]))

    return correct


if __name__ == '__main__':
    print(testIfGoLCorrect(gameOfLife, solutions))
