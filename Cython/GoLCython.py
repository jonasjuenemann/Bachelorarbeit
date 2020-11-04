import GoL
import numpy as np
from time import time

np.random.seed(0)
iterations = 200
N = 512
grid = np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N)
print(grid)

t_start = time()

grid = GoL.gameOfLife(grid, iterations)


print(grid)
t_end = time()

print("end")
print('Total time: %f' % (t_end - t_start))

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