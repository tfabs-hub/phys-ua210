import numpy as np
import matplotlib.pyplot as plt

# Define the wave function of a quantum harmonic oscillator
def psi(n, x):
    return ((1.0 / (np.sqrt((2**n) * np.math.factorial(n) * np.sqrt(np.pi)))) * np.exp(-x**2 / 2) * np.polynomial.hermite.hermval(x, [0]*n + [1]))

def integrand(x, n):
    return x**2 * (abs(psi(n, x)))**2

def integration(n):
    x = np.linspace(-10, 10, 1000)
    y = integrand(x, n)
    dx = x[1] - x[0]
    return np.sum(y*dx)

def calculate_uncertainty(n):
    return np.sqrt(integration(n))

n = 5
print("The uncertainty for n = 5 is approximately", calculate_uncertainty(n))

x = np.linspace(-10, 10, 1000)
y = psi(n, x)
plt.plot(x, y)
plt.title('Wave function for n = 5')
plt.xlabel('x')
plt.ylabel('psi_n(x)')
plt.show()