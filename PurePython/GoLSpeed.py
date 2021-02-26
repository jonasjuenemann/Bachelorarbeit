from time import time
from timeit import Timer
import numpy as np
import sys
from GoLFunctions import gameOfLife  , gameOfLifeNaiv
from GoLFunctionsJiT import gameOfLifeJiT
from GoLPara import gameOfLifePara

# np.set_printoptions(threshold=sys.maxsize)

X = [32, 64, 128, 256, 512, 1024, 2048]
Y = [1024, 2048, 4096, 8192, 16384]
iterations = 20

def gameOfLifeTimer(func, x , N):
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
    for i in range(x):
        grid = func(grid)

if __name__ == '__main__':

    for i in Y[-1:]:

        print("Durchführung mit Arraygröße: " + str(i))
        """
        t = Timer(lambda: gameOfLifeTimer(gameOfLifeNaiv, iterations,N))
        avgtime = t.timeit(number=1)

        print("end")
        print('Total time gameOfLifeNaiv: %f' % (avgtime))

        t = Timer(lambda: gameOfLifeTimer(gameOfLife, iterations,N))
        avgtime = t.timeit(number=1)

        print("end")
        print('Total time gameOfLife klassisch: %f' % (avgtime))
        """
        t = Timer(lambda: gameOfLifeTimer(gameOfLifeJiT, iterations, i))
        avgtime = t.timeit(number=1)

        print("end")
        print('Total time gameOfLife gameOfLifeJiT: %f' % (avgtime))

        t = Timer(lambda: gameOfLifeTimer(gameOfLifePara, iterations, i))
        avgtime = t.timeit(number=1)

        print("end")
        print('Total time JiT, parallel: %f' % (avgtime))


"""
t_start = time()
gameOfLifeTimer(gameOfLife, iterations,N)

t_end = time()
print("end")
print('Total time: %fs' % (t_end - t_start))

t_start = time()
gameOfLifeTimer(gameOfLife, iterations, N)

t_end = time()
print("end")
print('Total time 2nd: %fs' % (t_end - t_start))

t_start = time()
gameOfLifeTimer(gameOfLife, iterations, N)

t_end = time()
print("end")
print('Total time 3rd: %fs' % (t_end - t_start))
"""