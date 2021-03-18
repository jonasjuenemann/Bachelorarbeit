import matplotlib
from matplotlib import animation
import numpy as np
from GoL import GoL as gameOfLife

"""Bei mir Nötig, um manuelle Kontrolle über die Animation zu erlauben
Ist aber spezifisch, andere scheinen das nicht zu brauchen"""
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

"""Breite des Arrays"""
N = 128
"""Erstellung des Arrays"""
grid = np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N)
"""Erstellung der Subplots"""
fig, ax = plt.subplots()
"""Erstellung der ImageShow"""
img = ax.imshow(grid, interpolation='nearest')

"""
Dieser Funktion wird matplotlib spezifischer Paramter FrameNum übergeben.
Dieser braucht nicht weiter beachtet, ohne funktioniert die Funktionsanimation nicht.
außerdem wird der Funktion eine Imshow, sowie ein 2D-Array übergeben.
Die Funktion führt dann das Game Of Life auf dem übergeben Array aus.
Anschließend die Imshow auf das aktualisierte Array gesetzt.
Die Imshow wird anschließend zurückgegeben
"""


def update_grid(frameNum, img, grid_in):
    grid[:] = gameOfLife(grid_in)[:]
    img.set_data(grid)
    return img


"""
Mit der update_grid Funktion kann eine Funktionsanimation vorgenommen werden.
Dieser muss ein figure Objekt sowie eine aufrufbare Funktion übergeben werden.
Für diese können die Funktionsargumente definiert werden. Außerdem kann das Interval
zwischen den (in diesem Fall) Imageshows definiert werden.
Für die genauen Parameter kann ansonsten die matplotlib Dokumentation
https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.animation.FuncAnimation.html
konsultiert werden.
"""
ani = animation.FuncAnimation(fig, update_grid, fargs=(img, grid,), interval=25,
                              frames=1000, save_count=1000)
plt.show()
