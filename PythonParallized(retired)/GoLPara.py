import numpy as np
from time import time
# import sys
from joblib import Parallel, delayed, parallel_backend, Memory, dump, load
# np.set_printoptions(threshold=sys.maxsize)
import multiprocessing
import os
num_cores = multiprocessing.cpu_count()
"""funktioniert nicht so wirklich, wird nach der ersten Speicherung nicht mehr aktualisiert."""


# fuer @memory.cache

# memory = Memory(cachedir, verbose=0)
# GameOfLife = memory.cache(gameOfLife)

def findNew(y):
    N = grid.shape[1]
    array = np.full(N, 0, dtype=np.int8)
    # print("Working on x[" + str(x) + "] in row " + str(y) + ".")
    for x in range(N):
        z = grid[(y - 1 + N) % N][(x + 1 + N) % N] + grid[(y - 1 + N) % N][(x + N) % N] \
            + grid[(y - 1 + N) % N][(x - 1 + N) % N] + grid[(y + N) % N][(x + 1 + N) % N] + \
            grid[(y + N) % N][(x - 1 + N) % N] + grid[(y + 1 + N) % N][(x + 1 + N) % N] + \
            grid[(y + 1 + N) % N][(x + N) % N] + grid[(y + 1 + N) % N][(x - 1 + N) % N]
        if z == 3:
            array[x] = 1
            continue
        if (grid[y][x] == 1) and (z == 2):
            array[x] = 1
            continue
    return array


"""
def gameOfLife():
    global grid
    grid_out = np.zeros_like(grid)
    Size = len(grid)
    for y in range(Size):
        # print("y = " + str(y))
        #t_start = time()
        grid_out[y] = Parallel(num_cores-1)(delayed(findNew)(x, y) for x in range(Size)) # so benutzt er nur jeweils die Breite parallel.
        # braucht sehr lange, da der erste Aufruf sehr lang dauert und offensichtlich der Overhead bei sovielen Aufrufen viel zu hoch ist.
        #t_end = time()
        #print('Paralleler Aufruf fuer ' + str(y) + ' braucht %fs' % (t_end - t_start))
        # print("x = " + str(x))

    return grid_out
"""

"""
Because the proposed step with joblib creates that many full-scale process copies - so as to escape from the GIL-stepped pure-[SERIAL] dancing 
( one-after-another ) but (!) this includes add-on costs of all the memory transfers ( very expensive / sensitive for indeed large numpy arrays ) 
of all variables and the whole python interpreter and its internal state, before it gets to start doing a first step on the "usefull" work on your 
"payload"-calculation strategy,

Dispatching overhead: it is important to keep in mind that dispatching an item of the for loop has an overhead (much bigger than iterating a for loop 
without parallel). Thus, if these individual computation items are very fast, this overhead will dominate the computation. In the latest joblib, 
joblib will trace the execution time of each job and start bunching them if they are very fast. This strongly limits the impact of the dispatch 
overhead in most cases (see the PR for bench and discussion).

If you use the multiprocessing backend, then you have to copy the input data into each of four or eight processes (one per core), do the processing in 
each processes, and then copy the output data back. The copying is going to be slow, but if the processing is a little bit more complex than just 
calculating a square, it might be worth it. Measure and see.

"Many small tasks" are not a good fit for joblib. The coarser the task granularity, the less overhead joblib causes and the more benefit you will have from it. 
With tiny tasks, the cost of setting up worker processes and communicating data to them will outweigh any any benefit from parallelization.

Since numpy array operations are mostly by element it seems possible to parallelize them. But this would involve setting up either a shared memory 
segment for python objects, or dividing the arrays up into pieces and feeding them to the different processes, not unlike what multiprocessing.Pool 
does. No matter what approach is taken, it would incur memory and processing overhead to manage all that.

The slow_mean function introduces a time.sleep() call to simulate a more expensive computation cost for which parallel computing is beneficial. Parallel 
may not be beneficial for very fast operation, due to extra overhead (workers creations, communication, etc.).
"""


# Bringt kein Speedup, da nach jeder Funktion der Output in den Cache gelegt wird. Das nutzt nur nichts, da der naechste Aufurf ja mit anderem Wert ist.
# @memory.cache

# Auch eine Memmap zu benutzen, bei der man nach der Funktion das grid dumpt und bei Aufruf neu laedt bringt nichts, deutlich langsamer, ist ja auch recht offensichtlich
def gameOfLifePara(grid):
    # grid = load(data_GoL_memmap, mmap_mode='r')
    Size = grid.shape[0]
    # print("start")
    # t_start = time()
    """
    # Kein großer Unterschied:
    # with parallel_backend("loky", inner_max_num_threads=1):
    #    grid_out = Parallel(num_cores - 1, verbose=50)(delayed(findNew)(x, y) for y in range(Size) for x in range(Size))
    # sehr langsam ((~35 zu 4 sec)
    # grid_out = Parallel(num_cores, verbose=0, require='sharedmem')(delayed(findNew)(x, y) for y in range(Size) for x in range(Size))
    # threading scheint auch keine Option zu sein (auch she langsam) (~35 zu 4 sec)
    # Parallelisierung bringt schon Vorteile (num_cores=4 -> 4, sec., bei num_cores=1 -> 27)
    """
    grid_out = np.empty_like(grid)
    grid_out = Parallel(num_cores, verbose=0)(delayed(findNew)(y) for y in range(Size))

    # t_end = time()
    # print('Total time in parallel execution: %f' % (t_end - t_start))
    # print("start")
    # t_start = time()
    grid_out = np.array(grid_out).reshape(Size, Size)  # np.array() sowie .reshape jeweils O(n) time complexity hinzu, ist aber tatsaechlich, da sehr optimiert, sehr schnell, -> 0001 sec pro run.
    # t_end = time()
    # print('Total time Resizing: %f' % (t_end - tglobal grid_start))
    # dump(grid, data_GoL_memmap)
    return grid_out


if __name__ == '__main__':
    np.random.seed(0)
    iterations = 20
    N = 256
    grid = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
    # cachedir = r"C:\Users\JonasJuenemann\AppData\Local\cache"
    # path = r'C:\Users\JonasJuenemann\Documents\GitHub\Bachelorarbeit\PythonParallized(retired)\joblib_memmap'
    """
    # fuer memmapping
    folder = './joblib_memmap'
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    data_GoL_memmap = os.path.join(folder, 'data_memmap')
    # print(data_GoL_memmap) -> ./joblib_memmap\data_memmap
    dump(grid, data_GoL_memmap)
    # print(load(data_GoL_memmap, mmap_mode='r'))
    """
    print("started with gridsize " + str(N) + " and " + str(iterations) + " iterations")
    print(grid)
    t_start = time()
    # print(load(data_GoL_memmap, mmap_mode='r'))
    # Hier kann man nicht parallelisieren (der 2. Aufruf geht ja erst wenn der erste fertig ist.)
    for i in range(iterations):
        grid = gameOfLifePara(grid)
        #print(grid)
    # print(load(data_GoL_memmap, mmap_mode='r'))
    # print(grid)
    # timeit.Timer(gameOfLife(grid)).timeit(number=1
    t_end = time()
    print("end")
    print('Total time: %f' % (t_end - t_start))

"""

Das Ganze bringt aktuell nicht wirklich eine schnellere Ausfuehrung, langsamer fuer kleine Arrays bzw. eine Iteration, danach ungefaehr gleich schnell.
Wenig befriedigend.
Stack Overflow benutzt hier numba jit
Außerdem joblib mit numpy(?) vllt. mit besserem memory management doch speedup moeglich



"""
