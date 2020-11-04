import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""
PurePython
Cython
PybindsArrays
PybindsVektoren
C++
PyCuda
"""
fig = plt.figure(figsize=(20, 10))
Arraysizes = [32, 64, 128, 256, 512, 1024, 2048]
font = {'family': 'sans-serif',
        'color': 'blue',
        'weight': 'normal',
        'size': 16,
        }

# Iterations = 1;
# N=32 N=64 N=128 N=256 N=512 N=1024 N=2048
PythonNaiv = [0, 0.015, 0.093735, 0.343742, 1.343719, 5.468748, 21.796871]
PythonOpt = [0, 0.015, 0.046874, 0.296836, 1.126185, 4.640658, 17.562500]
# PythonParallel = [0.437467, 0.484343, 0.499969, 0.687502, 0.968747, 6.9, 26.802776]
PythonParallel = [1.21, 1.201, 1.243, 1.18, 1.177, 1.348, 1.437]
PythonNumbaJiT = [0.390, 0.406, 0.390, 0.437, 0.468, 0.562, 0.718]
Cython = [0.003, 0.01, 0.04, 0.165, 0.72, 2.778, 10.043]
PybindsArrays = [0, 0.003, 0.01, 0.042, 0.164, 0.67, 2.69]
PybindsVektoren = [0.001, 0.002, 0.003, 0.014, 0.053, 0.255, 0.83]
PybindsVParallel = [0, 0.002, 0.003, 0.011, 0.045, 0.231, 0.696]
Cplusplus = [0, 0.001, 0.002, 0.007, 0.027, 0.104, 0.415]
CplusplusParallel = [0.001, 0.002, 0.003, 0.006, 0.026, 0.083, 0.321]
PyCuda = [0, 0, 0, 0, 0.009, 0.033, 0.15]
NumbaCuda = [0.671, 0.671, 0.671, 0.671, 0.718, 0.671, 0.749]
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(Arraysizes, PythonNaiv, marker="o", c="yellow", markersize=8)
ax1.plot(Arraysizes, PythonOpt, c="green")
ax1.plot(Arraysizes, PythonParallel, c="red")
ax1.plot(Arraysizes, PythonNumbaJiT, c="black")
ax1.plot(Arraysizes, Cython, c="purple")
ax1.plot(Arraysizes, PybindsArrays)
ax1.plot(Arraysizes, PybindsVektoren)
# ax1.scatter(Arraysizes[4], PybindsVektoren[4], c="purple", marker="X", s=200)
ax1.plot(Arraysizes, PybindsVParallel)
ax1.plot(Arraysizes, Cplusplus, c="green")
ax1.plot(Arraysizes, CplusplusParallel)
ax1.plot(Arraysizes, PyCuda, marker="v", c="red", markersize=8)
ax1.plot(Arraysizes, NumbaCuda)
ax1.legend(["PythonNaiv", "PythonOpt", "PythonParallel", "PythonNumbaJiT", "Cython", "PybindsArrays", "PybindsVektoren",
            "PybindsVParallel", "C++", "PyCuda", "NumbaCuda"], loc=2, prop={'size': 16})
# x_ticks = np.append(ax1.get_xticks(), Arraysizes)
ax1.axis([0, 2100, 0, 25])
ax1.set_xticks(Arraysizes)
ax1.set_xticklabels(Arraysizes, rotation=60)
ax1.set_title("GoL speed of Execution for 1 Iteration", fontdict=font)
ax1.set_xlabel("Arraysize", fontdict=font)
ax1.set_ylabel("Execution time", fontdict=font)

plt.savefig("Alles1.jpg")

fig = plt.figure(figsize=(20, 10))
bestpy = fig.add_subplot(1, 1, 1)
bestpy.plot(Arraysizes, PythonNaiv, marker="o", c="yellow", markersize=8)
bestpy.plot(Arraysizes, PythonOpt, c="green")
bestpy.plot(Arraysizes, Cython, c="purple")
bestpy.plot(Arraysizes, PythonParallel, c="red")
bestpy.plot(Arraysizes, PythonNumbaJiT, c="black")
bestpy.legend(["PythonNaiv", "PythonOpt", "Cython", "PythonParallel", "PythonNumbaJiT"], loc=2, prop={'size': 16})
bestpy.axis([0, 2100, 0, 25])
bestpy.set_xticks(Arraysizes)
bestpy.set_xticklabels(Arraysizes, rotation=60)
bestpy.set_title("GoL speed of Execution for 1 Iteration", fontdict=font)
bestpy.set_xlabel("Arraysize", fontdict=font)
bestpy.set_ylabel("Execution time", fontdict=font)

plt.savefig("BestPython1.jpg")

# Iterations = 20;
# N=32 N=64 N=128 N=256 N=512 N=1024 N=2048
PythonNaiv = [0.093, 0.421, 1.625, 7.015, 29.481, 106.79, 420]
PythonOpt = [0.078, 0.312, 1.296, 5.343, 24.412, 88.109, 340]
# PythonParallel = [0.765, 1.109, 1.765, 3.734, 12.562, 50.341, 555.188]
PythonParallel = [1.120, 1.098, 1.237, 1.326, 1.383, 1.881, 3.257]
PythonNumbaJiT = [0.437, 0.437, 0.453, 0.703, 0.825, 1.438, 3.422]
Cython = [0.066, 0.261, 0.908, 3.214, 14.207, 48.58, 196.6]
PybindsArrays = [0.013, 0.052, 0.201, 0.83, 3.447, 13.6, 54.202]
PybindsVektoren = [0.001, 0.006, 0.02, 0.076, 0.326, 1.25, 4.984]
PybindsVParallel = [0.001, 0.004, 0.014, 0.027, 0.18, 0.47, 1.89]
Cplusplus = [0.001, 0.002, 0.009, 0.036, 0.139, 0.567, 2.32]
CplusplusParallel = [0.001, 0.002, 0.006, 0.02, 0.073, 0.227, 0.862]
PyCuda = [0, 0, 0, 0, 0.016, 0.031, 0.141]
NumbaCuda = [0.687, 0.671, 0.749, 0.749, 0.771, 0.771, 0.828]
fig = plt.figure(figsize=(20, 10))
ax3 = fig.add_subplot(1, 1, 1)
ax3.plot(Arraysizes, PythonNaiv, marker="o", c="yellow", markersize=8)
ax3.plot(Arraysizes, PythonOpt, c="orange")
ax3.plot(Arraysizes, PythonParallel, c="yellowgreen")
ax3.plot(Arraysizes, PythonNumbaJiT, c="purple")
ax3.plot(Arraysizes, Cython, c="blue")
ax3.plot(Arraysizes, PybindsArrays)
ax3.plot(Arraysizes, PybindsVektoren)
# ax1.scatter(Arraysizes[4], PybindsVektoren[4], c="purple", marker="X", s=200)
ax3.plot(Arraysizes, PybindsVParallel)
ax3.plot(Arraysizes, Cplusplus, c="green")
ax3.plot(Arraysizes, CplusplusParallel)
ax3.plot(Arraysizes, PyCuda, marker="v", c="red", markersize=8)
ax3.plot(Arraysizes, NumbaCuda)
ax3.legend(["PythonNaiv", "PythonOpt", "PythonParallel", "PythonNumbaJiT", "Cython", "PybindsArrays", "PybindsVektoren",
            "PybindsVParallel", "C++", "PyCuda", "NumbaCuda"], loc=2, prop={'size': 8})
# x_ticks = np.append(ax1.get_xticks(), Arraysizes)
ax3.axis([0, 2100, 0, 210])
ax3.set_xticks(Arraysizes)
ax3.set_xticklabels(Arraysizes, rotation=60)
ax3.set_title("GoL speed of Execution for 20 Iterations", fontdict=font)
ax3.set_xlabel("Arraysize", fontdict=font)
ax3.set_ylabel("Execution time", fontdict=font)
plt.savefig("Alles20.jpg")
plt.show()

fig = plt.figure(figsize=(20, 10))
bestpy = fig.add_subplot(1, 1, 1)
bestpy.plot(Arraysizes, PythonNaiv, marker="o", c="yellowgreen", markersize=8)
bestpy.plot(Arraysizes, PythonOpt, c="green")
bestpy.plot(Arraysizes, Cython, c="purple")
bestpy.plot(Arraysizes, PythonParallel, c="black")
bestpy.plot(Arraysizes, PythonNumbaJiT, c="red")
bestpy.plot(Arraysizes, PybindsArrays)
bestpy.plot(Arraysizes, PybindsVektoren)
bestpy.plot(Arraysizes, PybindsVParallel)
bestpy.legend(["PythonNaiv", "PythonOpt", "Cython", "PythonParallel", "PythonNumbaJiT", "PybindsArrays", "PybindsVektoren",
            "PybindsVParallel"], loc=2, prop={'size': 16})
bestpy.axis([0, 2100, 0, 310])
bestpy.set_xticks(Arraysizes)
bestpy.set_xticklabels(Arraysizes, rotation=60)
bestpy.set_title("GoL speed of Execution for 20 Iterations", fontdict=font)
bestpy.set_xlabel("Arraysize", fontdict=font)
bestpy.set_ylabel("Execution time", fontdict=font)

plt.savefig("Python&C20.jpg")


# N=32 N=64 N=128 N=256 N=512 N=1024 N=2048
PythonNaiv = [1.079, 4.319, 17.134, 71.215, 305.977, 1300, 5000]
PythonOpt = [0.924, 3.56, 13.93, 54.94, 210, 850, 3200]
PythonParallel = [1.148, 1.283, 1.345, 1.549, 2.57, 5.927, 20.048]
PythonNumbaJiT = [0.373, 0.395, 0.450, 0.726, 1.968, 6.778, 26.775]
Cython = [0.534, 2.203, 7.833, 32.102, 123.689, 480, 1920]
PybindsArrays = [0.125, 0.496, 2.198, 8.457, 34.204, 130.224, 520]
PybindsVektoren = [0.01, 0.047, 0.182, 0.678, 3.22, 11.59, 43.483]
PybindsVParallel = [0.006, 0.019, 0.095, 0.334, 0.92, 3.25, 14.09]
Cplusplus = [0.004, 0.018, 0.078, 0.293, 1.132, 4.748, 19.558]
CplusplusParallel = [0.004, 0.015, 0.041, 0.101, 0.422, 1.961, 6.331]
PyCuda = [0.016, 0.016, 0.016, 0.016, 0.031, 0.062, 0.235]
NumbaCuda = [0.734, 0.703, 0.734, 0.687, 0.765, 0.859, 1.203]
fig = plt.figure(figsize=(20, 10))
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(Arraysizes, PythonNaiv, marker="o", c="yellow", markersize=8)
# ax2.scatter(Arraysizes[4], PythonNaiv[4], c="yellow", marker="X", s=200)
ax2.plot(Arraysizes, PythonOpt)
ax2.plot(Arraysizes, PythonParallel)
ax2.plot(Arraysizes, PythonNumbaJiT)
ax2.plot(Arraysizes, Cython, c="orange")
ax2.plot(Arraysizes, PybindsArrays, c="blue")
ax2.plot(Arraysizes, PybindsVektoren, c="purple")
ax2.plot(Arraysizes, PybindsVParallel)
ax2.plot(Arraysizes, Cplusplus, marker="D", c="green")
ax2.plot(Arraysizes, CplusplusParallel)
ax2.plot(Arraysizes, PyCuda, marker="v", c="red", markersize=8)
ax2.plot(Arraysizes, NumbaCuda)
ax2.legend(["PythonNaiv", "PythonOpt", "PythonParallel", "PythonNumbaJiT", "Cython", "PybindsArrays", "PybindsVektoren",
            "PybindsVParallel", "C++", "PyCuda", "NumbaCuda"], prop={'size': 16})
# x_ticks = np.append(ax1.get_xticks(), Arraysizes)
ax2.axis([0, 2100, 0, 310])
ax2.set_xticks(Arraysizes)
ax2.set_xticklabels(Arraysizes, rotation=60)
ax2.set_title("GoL speed of Execution for 200 Iterations", fontdict=font)
ax2.set_xlabel("Arraysize", fontdict=font)
ax2.set_ylabel("Execution time", fontdict=font)
plt.savefig("Alles200.jpg")
plt.show()

fig = plt.figure(figsize=(20, 10))
bestpy = fig.add_subplot(1, 1, 1)
bestpy.plot(Arraysizes, PythonNaiv, marker="o", c="yellow", markersize=8)
bestpy.plot(Arraysizes, PythonOpt, c="green")
bestpy.plot(Arraysizes, Cython, c="purple")
bestpy.plot(Arraysizes, PythonParallel, c="black")
bestpy.plot(Arraysizes, PythonNumbaJiT, c="red")
bestpy.legend(["PythonNaiv", "PythonOpt", "Cython", "PythonParallel", "PythonNumbaJiT"], loc=2, prop={'size': 16})
bestpy.axis([0, 2100, 0, 310])
bestpy.set_xticks(Arraysizes)
bestpy.set_xticklabels(Arraysizes, rotation=60)
bestpy.set_title("GoL speed of Execution for 200 Iterations", fontdict=font)
bestpy.set_xlabel("Arraysize", fontdict=font)
bestpy.set_ylabel("Execution time", fontdict=font)

plt.savefig("Python200.jpg")

fig = plt.figure(figsize=(20, 10))
bestpy = fig.add_subplot(1, 1, 1)
bestpy.plot(Arraysizes, Cython, c="purple")
bestpy.plot(Arraysizes, PythonParallel, c="black")
bestpy.plot(Arraysizes, PythonNumbaJiT, c="red")
bestpy.plot(Arraysizes, PybindsVektoren)
bestpy.plot(Arraysizes, PythonParallel, c="black")
bestpy.plot(Arraysizes, PybindsVParallel)
bestpy.legend(["Cython", "PythonParallel", "PythonNumbaJiT", "PybindsVektoren",
            "PybindsVParallel"], loc=2, prop={'size': 16})
bestpy.axis([0, 2100, 0, 50])
bestpy.set_xticks(Arraysizes)
bestpy.set_xticklabels(Arraysizes, rotation=60)
bestpy.set_title("GoL speed of Execution for 20 Iterations", fontdict=font)
bestpy.set_xlabel("Arraysize", fontdict=font)
bestpy.set_ylabel("Execution time", fontdict=font)

plt.savefig("BestPythonVSC-Interface.jpg")

# Iterations = 200;


# Nur fuer die fixen:
# Iterations = 200;
Arraysizes = [1024, 2048, 4096, 8192, 16384]
# N=1024 N=2048 N=4096 N=8192 N=16384
PythonParallel = [5.927, 20.048, 81.433, 320, 1280]
PythonNumbaJiT = [6.778, 26.775, 102.99, 410, 1640]
PybindsVParallel = [3.25, 14.09, 53.221, 212, 840]
CplusplusParallel = [1.961, 6.331, 20.033, 76.73, 310]
PyCuda = [0.062, 0.235, 0.969, 3.657, 14.828]
NumbaCuda = [0.859, 1.203, 2.968, 9.828, 37.328]
fig = plt.figure(figsize=(20, 10))
ax4 = fig.add_subplot(1, 1, 1)
ax4.axis([1000, 16500, 0, 800])
ax4.plot(Arraysizes, PythonParallel, marker="o", c="black", markersize=8)
ax4.plot(Arraysizes, PythonNumbaJiT)
ax4.plot(Arraysizes, PybindsVParallel)
ax4.plot(Arraysizes, CplusplusParallel, c="blue")
ax4.plot(Arraysizes, PyCuda, marker="v", c="red", markersize=8)
ax4.plot(Arraysizes, NumbaCuda)
ax4.legend(["PythonParallel", "PythonNumbaJiT", "PybindsVParallel", "CplusplusParallel", "PyCuda", "NumbaCuda"], loc=2,
           prop={'size': 16})
ax4.set_xticks(Arraysizes)
ax4.set_xticklabels(Arraysizes, rotation=60)
ax4.set_title("GoL speed of Execution for 200 Iterations", fontdict=font)
ax4.set_xlabel("Arraysize", fontdict=font)
ax4.set_ylabel("Execution time", fontdict=font)

# plt.subplots_adjust(hspace=0.3, bottom=0.05, top=0.95, right=0.9, left=0.1)
plt.savefig("Top200.jpg")

"""
Compilation time vs Runtime wurde hier jetzt nicht gemacht.
"""
