from math import sqrt
from time import time

from joblib import Parallel, delayed, parallel_backend, Memory
cachedir = r'C:\Users\JonasJuenemann\AppData\Local\cache'
memory = Memory(cachedir, verbose=0)
@memory.cache
def f(x):
    print('Running f(%s)' % x)
    return x

if __name__ == '__main__':
    t_start = time()
    a = [sqrt(i ** 2) for i in range(1000)]
    t_end = time()
    print("end")
    print('Total time: %f' % (t_end - t_start))

    t_start = time()
    #b = Parallel(n_jobs=2,)(delayed(sqrt)(i ** 2) for i in range(10000000))
    t_end = time()
    print("end")
    print('Total time Parallel: %f' % (t_end - t_start))

    print(f(1))
    print(f(1))
    x = 1
    print(f(x))
    print(f(x))
    x = 2
    print(f(x))


