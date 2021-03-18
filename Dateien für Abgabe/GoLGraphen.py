import matplotlib.pyplot as plt

"""Erstellung der Graphen"""

"""Einstellungen bezüglich des Texts"""
font = {'family': 'sans-serif',
        'color': 'blue',
        'weight': 'normal',
        'size': 24,
        }

"""Einstellungen bezüglich der Größe Ticks sowie der Graphen selbst"""
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('lines', linewidth=3.0)

Arraysizes = [32, 64, 128, 256, 512, 1024, 2048]

"""Ergebnisse für die Werkzeuge aus den Performance-Tests"""
PythonNaiv = [0.083, 0.341, 1.367, 5.853, 30.837, 102.556, 356.018]
PythonOpt = [0.061, 0.257, 1.059, 4.358, 19.311, 83.408, 300]
PythonNumbaJiT = [0.004, 0.006, 0.011, 0.033, 0.140, 0.557, 1.911]
PythonNumbaParallel = [0.010, 0.012, 0.017, 0.036, 0.142, 0.437, 1.711]
Cython = [0.028, 0.106, 0.455, 1.69, 6.93, 27.62, 121.369]
PybindsArrays = [0.013, 0.052, 0.201, 0.83, 3.447, 13.6, 54.202]
PybindsVektoren = [0.001, 0.006, 0.02, 0.076, 0.326, 1.25, 4.984]
PybindsVParallel = [0.0008, 0.0098, 0.014, 0.038, 0.15, 0.434, 1.76]
Cplusplus = [0.0005, 0.00165, 0.008, 0.03, 0.106, 0.441, 1.68]
CplusplusParallel = [0.0005, 0.0008, 0.0021, 0.0078, 0.0405, 0.188, 0.5347]
PyCuda = [0.0017, 0.0018, 0.0020, 0.0029, 0.01064, 0.03632, 0.15034]
NumbaCuda = [0.016, 0.009, 0.009, 0.010, 0.021, 0.053, 0.200]

"""Erstellung eines Graphen am Beispiel des ersten Python-Graphs"""
fig = plt.figure(figsize=(20, 10))
bestpy = fig.add_subplot(1, 1, 1)
bestpy.plot(Arraysizes, PythonNaiv, c="yellowgreen", markersize=8)
bestpy.plot(Arraysizes, PythonOpt, c="green")
bestpy.legend(["PythonNaiv", "PythonOpt"], loc=2, prop={'size': 32})
bestpy.axis([0, 2100, 0, 370])
bestpy.set_xticks(Arraysizes)
bestpy.set_xticklabels(Arraysizes, rotation=60)
bestpy.set_title("Geschwindigkeit des GoL über 20 Iterationen", fontdict=font)
bestpy.set_xlabel("Arraygröße", fontdict=font)
bestpy.set_ylabel("Ausführungszeit in Sekunden", fontdict=font)
"""Speichern des Graphs als jpg"""
plt.savefig("JustPython.jpg")

"""Ergebnis über große Arrays mit den schnelleren Varianten"""
Arraysizes = [1024, 2048, 4096, 8192, 16384]
PythonParallel = [0.437, 1.711, 7.691, 25.971, 103.51]
CplusplusParallel = [0.188, 0.5347, 2.15, 8.05, 30]
PyCuda = [0.0372, 0.142, 0.55, 2.193, 8.972]
NumbaCuda = [0.053, 0.200, 0.733, 2.876, 11.872]

"""Erstellung eines Graphen am Beispiel des letzten Graphen"""
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(Arraysizes, PythonParallel, c="yellow")
ax1.plot(Arraysizes, PyCuda, c="black")
ax1.plot(Arraysizes, NumbaCuda, c="red")
ax1.plot(Arraysizes, CplusplusParallel, c="blue")
ax1.legend(["PythonNumbaParallel", "PyCUDA", "NumbaCUDA", "Cplusplus"], loc=2, prop={'size': 32})
ax1.axis([0, 2100, 0, 90])
ax1.set_xticks(Arraysizes)
ax1.set_xticklabels(Arraysizes, rotation=60)
ax1.set_title("Geschwindigkeit des GoL über 20 Iterationen", fontdict=font)
ax1.set_xlabel("Arraygröße", fontdict=font)
ax1.set_ylabel("Ausführungszeit in Sekunden", fontdict=font)
"""Speichern des Graphs als jpg"""
plt.savefig("numbaCuda&PyCuda&C++&numbaPython.jpg")
