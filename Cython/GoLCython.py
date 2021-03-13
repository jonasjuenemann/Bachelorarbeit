import numpy as np
from time import time
from timeit import Timer
import GameOfLifeCython

np.random.seed(0)
iterations = 20
N = [32, 64, 128, 256, 512]
"1024, 2048"

def gameOfLifeTimer(func, x , N):
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
    grid = func(grid, x)

for i in N:
    print("Durchführung mit Arraygröße: " + str(i))

    t = Timer(lambda: gameOfLifeTimer(GameOfLifeCython.gameOfLife, iterations, i))
    avgtime = t.timeit(number=3) / 3

    print("end")
    print('Total time gameOfLife via Cython: %f' % (avgtime))



"""
python 3.8.5 distri
it=1
N=256
Total time: 0.166998

it=10
N=256
Total time: 1.702709


python ML venv
it=1
N=256
Total time: 0.347892
"""