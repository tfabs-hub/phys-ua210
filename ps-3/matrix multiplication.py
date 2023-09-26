
import numpy as np
import time
import matplotlib.pyplot as plt

def matrix_mult(A, B):
    n = len(A)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

sizes = np.arange(10, 110, 10)
times_explicit = []
times_dot = []

for N in sizes:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)

    start = time.time()
    matrix_mult(A, B)
    end = time.time()
    times_explicit.append(end - start)

    start = time.time()
    np.dot(A, B)
    end = time.time()
    times_dot.append(end - start)

plt.figure(figsize=(10, 5))
plt.plot(sizes, times_explicit, label='Explicit function')
plt.plot(sizes, times_dot, label='np.dot() method')
plt.xlabel('Matrix size (N)')
plt.ylabel('Computation time (seconds)')
plt.legend()
plt.show()