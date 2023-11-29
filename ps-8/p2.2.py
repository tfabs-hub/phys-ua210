import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

sigma, beta, rho = 10, 8/3, 28
x0, y0, z0 = 0, 1, 0

tmax = 50
n = 10000

def lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations."""
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

t = np.linspace(0, tmax, n)
f = odeint(lorenz, (x0, y0, z0), t, args=(sigma, beta, rho))
x, y, z = f.T

plt.figure()
plt.plot(x, z)
plt.xlabel('x')
plt.ylabel('z')
plt.show()
