from time import time
import numpy as np
import sys
from GoLFunctions import gameOfLife, gameOfLifeNaiv
from GoLFunctionsJiT import gameOfLifeJit
from GoLPara import gameOfLifePara
# np.set_printoptions(threshold=sys.maxsize)
np.random.seed(0)
X = [32, 64, 128, 256, 512, 1024, 2048]
Y = [1024, 2048, 4096, 8192, 16384]


if __name__ == '__main__':
    iterations = 20
    N = X[1]
    print("started with gridsize " + str(N) + " and " + str(iterations) + " iterations")

    t_start = time()
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))

    for i in range(iterations):
        #grid = gameOfLifeNaiv(grid)
        pass

    t_end = time()
    # timeit.Timer(gameOfLife(grid)).timeit(number=1000)
    print("end")
    print('Total time W/ lists: %f' % (t_end - t_start))

    t_start = time()
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))

    for i in range(iterations):
        grid = gameOfLife(grid)
        pass

    t_end = time()
    print("end")
    print('Total time optimierter: %f' % (t_end - t_start))

    t_start = time()
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))

    for i in range(iterations):
        #grid = gameOfLifePara(grid)
        pass

    t_end = time()
    print("end")
    print('Total time parallel: %f' % (t_end - t_start))

    t_start = time()
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))

    for i in range(iterations):
        #grid = gameOfLifeJit(grid)
        pass


    t_end = time()
    print("end")
    print('Total time with jit: %f' % (t_end - t_start))

