import numpy as np
import matplotlib.pyplot as plt


def psi(n, x):
    return ((1.0 / (np.sqrt((2**n) * np.math.factorial(n) * np.sqrt(np.pi)))) * np.exp(-x**2 / 2) * np.polynomial.hermite.hermval(x, [0]*n + [1]))


def integrand(x, n):
    return x**2 * (abs(psi(n, x)))**2


def gauss_hermite_quad(n):
    x, w = np.polynomial.hermite.hermgauss(100)
    return np.sum(w * integrand(x, n))


def calculate_uncertainty(n):
    return np.sqrt(gauss_hermite_quad(n))


n = 5
print("The uncertainty for n = 5 is approximately", calculate_uncertainty(n))


x = np.linspace(-10, 10, 1000)
y = psi(n, x)
plt.plot(x, y)
plt.title('Wave function for n = 5')
plt.xlabel('x')
plt.ylabel('psi_n(x)')
plt.show()