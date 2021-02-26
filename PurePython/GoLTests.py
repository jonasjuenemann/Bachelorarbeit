from GoLFunctions import gameOfLife
from GoLFunctionsJiT import gameOfLifeJit
from GoLPara import gameOfLifePara
import numpy as np

np.set_printoptions(threshold=np.inf)

def testIfGoLCorrect(func):
    correct = True
    N = 10
    iterations = [1, 2]
    # Lösungen die sicher stimmen:
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

    for x in range(len(iterations)):
        np.random.seed(0)
        grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
        #print(grid)
        for i in range(iterations[x]):
            grid = func(grid)
            #print(grid)
        #print(np.alltrue(grid == solutions[x]))
        correct = (correct and np.alltrue(grid == solutions[x]))

    return correct

if __name__ == '__main__':
    print(testIfGoLCorrect(gameOfLife))
    print(testIfGoLCorrect(gameOfLifeJit))
    print(testIfGoLCorrect(gameOfLifePara))