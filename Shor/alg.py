import matplotlib.pyplot as plt
import math
import numpy as np

N = 21
m = 5
f = lambda x: m**x%N

print(f(np.arange(1,100)))

max = 1e3
start = 2
length = 0
for x in np.arange(2, max):
    if f(0+length) == f(x):
        length += 1
        if (start == length):
            break
    else:
        start = x+1

print('start', start)
print('length', length)