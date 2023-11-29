import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Lorenz paramters and initial conditions
sigma, beta, rho = 10, 8/3, 28
x0, y0, z0 = 0, 1, 0

# Maximum time point and total number of time points
tmax, n = 50, 10000

def lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations."""
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

# Integrate the Lorenz equations on the time grid t
t = np.linspace(0, tmax, n)
f = odeint(lorenz, (x0, y0, z0), t, args=(sigma, beta, rho))
x, y, z = f.T

# Plot y as a function of time
plt.figure()
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('y')
plt.show()
