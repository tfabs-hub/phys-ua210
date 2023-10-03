import numpy as np
import matplotlib.pyplot as plt

def H(n, x):
    """
    Calculate the Hermite polynomial Hn(x)
    """
    # Coefficients for Hermite polynomial of degree n
    coeff = np.zeros(n+1)
    coeff[n] = 1
    return np.polyval(coeff, x) * np.exp(-x**2 / 2) / np.sqrt(np.sqrt(np.pi) * 2**n * np.math.factorial(n))


"""
x = np.linspace(-4, 4, 1000)


for n in range(4):
    y = H(n, x)
    plt.plot(x, y, label=f'n = {n}')

plt.title('Harmonic Oscillator Wavefunctions')
plt.xlabel('x')
plt.ylabel('Wavefunction')
plt.legend()
plt.grid(True)
plt.show()"""

x = np.linspace(-10, 10, 1000)

# Calculate wavefunction for n = 30
y = H(30, x)

plt.figure(figsize=(10,5))
plt.plot(x, y)
plt.title('Harmonic Oscillator Wavefunction for n = 30')
plt.xlabel('x')
plt.ylabel('Wavefunction')
plt.grid(True)
plt.show()