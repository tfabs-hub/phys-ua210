import numpy as np
import matplotlib.pyplot as plt


def f(x, a):
    return (x**(a - 1)) * np.exp(-x)

x = np.linspace(0, 5, 100)

plt.figure(figsize=(10,6))
for a in [2, 3, 4]:
    plt.plot(x, f(x, a), label=f'a={a}')

plt.title('Graph of the integrand $(x^a-1)(e^{-x})$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()