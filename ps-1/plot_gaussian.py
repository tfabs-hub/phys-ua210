#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

mean = 0
std_dev = 3
x = np.linspace(-10, 10, 100)
y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

plt.plot(x, y)

plt.xlabel('x')
plt.ylabel('Probability Density')
plt.savefig("Gaussian")
plt.show()
