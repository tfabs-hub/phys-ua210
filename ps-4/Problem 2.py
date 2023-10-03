import numpy as np
import matplotlib.pyplot as plt



def V(x):
    return x**4

def integrand(x, a):
    return 1 / np.sqrt(V(a) - V(x))

def period(a, N=20):
    x, w = np.polynomial.legendre.leggauss(N)
    # Change of variables to compute integral from 0 to a
    x = 0.5 * (x + 1) * a
    w *= 0.5 * a
    return np.sqrt(8) * sum(w * integrand(x, a))


period_vec = np.vectorize(period)


a_values = np.linspace(0, 2, 100)
T_values = period_vec(a_values)


plt.figure(figsize=(10,5))
plt.plot(a_values, T_values)
plt.title('Period of an oscillator with potential V(x) = x^4')
plt.xlabel('Amplitude (a)')
plt.ylabel('Period (T)')
plt.grid(True)
plt.show()