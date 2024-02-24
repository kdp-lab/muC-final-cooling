# Quick test of multiprocessing correctness

import scan
import numpy as np
from time import sleep, time
from functools import partial


def f(x, y):
    sleep(x)
    return np.square(x)+np.square(y)


if __name__ == '__main__':
    test = np.array([1, 2, 4, 5])
    test_y = 5
    test_func = partial(f, y=test_y)
    start = time()
    print(scan.run_scan(test_func, (test,), processes=4).data)
    print(time() - start)
