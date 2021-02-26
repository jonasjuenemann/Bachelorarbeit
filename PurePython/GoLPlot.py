from GoLFunctions import gameOfLife
import numpy as np
import matplotlib
from matplotlib import animation
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

np.random.seed(0)
N = 10
grid = np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N)
fig, ax = plt.subplots()
img = ax.imshow(grid, interpolation='nearest')

def update_grid(frameNum, img, grid_in):
    grid[:] = gameOfLife(grid_in)[:]
    img.set_data(grid)
    return img

#ani = animation.FuncAnimation(fig, update_grid, fargs=(img, grid, ), interval=25,
#                                 frames=5, save_count=1000)
plt.show()